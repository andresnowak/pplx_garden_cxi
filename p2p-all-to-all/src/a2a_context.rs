use std::{env, ffi::c_void, ptr::null_mut, sync::Arc, thread::JoinHandle};

use anyhow::{Result, anyhow};
use cuda_lib::{
    CudaDeviceMemory, cuda_check, cudart_sys,
    rt::{CudartError, cudaGetNumSMs},
};
use fabric_lib::{TransferEngine, api::MemoryRegionHandle};
use thread_lib::pin_cpu;
use torch_lib::ScalarType;

use crate::{a2a_handles::AllToAllRankHandle, a2a_worker::WorkerState};

pub struct AllToAllDebugState {
    pub tokens_per_expert: Vec<u32>,
    pub token_offset: Vec<u32>,
    pub expert_offsets: Vec<u32>,
    pub combine_send_offset: Vec<u32>,
    pub source_dispatch_offset: Vec<u32>,
    pub source_rank: Vec<u32>,
    pub padded_index: Vec<u32>,
    pub num_recv_tokens: Vec<u32>,
    pub sum_tokens_per_expert: u32,
    pub num_recv_tokens_main: u32,
    pub num_recv_efa_tokens: u32,
    pub total_padded_tokens: u32,
    pub max_padded_index: Option<u32>,
    pub padded_index_out_of_bounds: usize,
}

// Collects the private workspace buffers used by dispatch and combine.
struct DeviceWorkspace {
    /// The offset of each expert within the contiguous token buffer.
    expert_offsets: CudaDeviceMemory,
    /// The offset of the token within the expert group.
    token_offset: CudaDeviceMemory,
    /// Counter for the number of tokens sent during combine.
    token_counter: CudaDeviceMemory,
    /// Counter for per-grid synchronization.
    grid_counter: CudaDeviceMemory,
    /// Counter for synchronization barriers across NVLink.
    sync_counter: CudaDeviceMemory,
    /// Device-side sync pointers.
    sync_ptrs: Option<CudaDeviceMemory>,
    /// Device-side send pointers.
    send_ptrs: Option<CudaDeviceMemory>,
    /// Device-side recv pointers.
    recv_ptrs: Option<CudaDeviceMemory>,
}

impl DeviceWorkspace {
    pub fn new(
        num_experts: usize,
        max_num_tokens: usize,
        num_experts_per_token: usize,
        host_sync_ptrs: &[u64],
        host_send_ptrs: &[u64],
        host_recv_ptrs: &[u64],
    ) -> Result<Self, CudartError> {
        let expert_offsets =
            CudaDeviceMemory::device(num_experts * std::mem::size_of::<u32>())?;
        expert_offsets.zero();

        let token_offset = CudaDeviceMemory::device(
            max_num_tokens * num_experts_per_token * std::mem::size_of::<u32>(),
        )?;

        let token_counter = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        token_counter.zero();
        let sync_counter = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        sync_counter.zero();
        let grid_counter = CudaDeviceMemory::device(std::mem::size_of::<u32>())?;
        grid_counter.zero();

        let sync_ptrs = if host_sync_ptrs.is_empty() {
            None
        } else {
            Some(CudaDeviceMemory::from_vec(host_sync_ptrs)?)
        };
        let send_ptrs = if host_send_ptrs.is_empty() {
            None
        } else {
            Some(CudaDeviceMemory::from_vec(host_send_ptrs)?)
        };
        let recv_ptrs = if host_recv_ptrs.is_empty() {
            None
        } else {
            Some(CudaDeviceMemory::from_vec(host_recv_ptrs)?)
        };

        Ok(Self {
            expert_offsets,
            token_offset,
            token_counter,
            grid_counter,
            sync_counter,
            sync_ptrs,
            send_ptrs,
            recv_ptrs,
        })
    }

    fn get_sync_ptr(&mut self) -> *mut *mut u32 {
        self.sync_ptrs.as_mut().map_or(null_mut(), |p| p.get_mut_ptr())
    }

    fn get_recv_ptr(&mut self) -> *mut *mut c_void {
        self.recv_ptrs.as_mut().map_or(null_mut(), |p| p.get_mut_ptr())
    }

    fn get_send_ptr(&mut self) -> *mut *mut c_void {
        self.send_ptrs.as_mut().map_or(null_mut(), |p| p.get_mut_ptr())
    }
}

#[allow(dead_code)]
pub struct AllToAllContext {
    hidden_dim: usize,
    hidden_dim_scale: usize,
    in_elemsize: usize,
    out_elemsize: usize,
    out_dtype: ScalarType,
    scale_elemsize: usize,
    num_experts: usize,
    max_num_tokens: usize,
    num_experts_per_token: usize,
    max_private_tokens: usize,
    expert_padding: usize,
    rank: usize,
    dp_size: usize,
    node_size: usize,
    world_size: usize,
    device: u8,
    workspace: DeviceWorkspace,
    worker: Arc<WorkerState>,
    thread: Option<JoinHandle<()>>,
    num_blocks: usize,
}

impl AllToAllContext {
    fn debug_dispatch_offsets_enabled() -> bool {
        env::var("PPLX_DEBUG_DISPATCH_OFFSETS").ok().as_deref() == Some("1")
    }

    fn debug_dispatch_memory_ranges_enabled() -> bool {
        env::var("PPLX_DEBUG_DISPATCH_MEMORY_RANGES").ok().as_deref() == Some("1")
    }

    fn debug_sync_dispatch_stream(stream: u64) -> Result<()> {
        if env::var("PPLX_DEBUG_SYNC_DISPATCH_SEND").ok().as_deref() != Some("1") {
            return Ok(());
        }

        let ret = unsafe {
            cudart_sys::cudaStreamSynchronize(stream as cudart_sys::cudaStream_t)
        };
        if ret != 0 {
            return Err(anyhow!(
                "cudaStreamSynchronize failed after dispatch_send: {}",
                ret
            ));
        }
        Ok(())
    }

    fn debug_print_dispatch_memory_ranges(&self, label: &str) {
        if !Self::debug_dispatch_memory_ranges_enabled() {
            return;
        }

        let expert_offsets_ptr = self.workspace.expert_offsets.ptr().as_ptr() as usize;
        let expert_offsets_size = self.workspace.expert_offsets.size();
        let token_offset_ptr = self.workspace.token_offset.ptr().as_ptr() as usize;
        let token_offset_size = self.workspace.token_offset.size();
        let send_buffer_ptr = self.worker.buffers.send_buffer_ptr as usize;
        let recv_buffer_ptr = self.worker.buffers.recv_buffer_ptr as usize;
        let tokens_per_expert_ptr = self.worker.tokens_per_expert.get_device_ptr() as usize;
        let source_dispatch_offset_ptr =
            self.worker.source_dispatch_offset.get_device_ptr() as usize;
        let combine_send_offset_ptr =
            self.worker.combine_send_offset.get_device_ptr() as usize;
        let source_rank_ptr = self.worker.source_rank.get_device_ptr() as usize;
        let padded_index_ptr = self.worker.padded_index.get_device_ptr() as usize;
        let num_recv_tokens_ptr = self.worker.num_recv_tokens.get_device_ptr() as usize;

        let tokens_per_expert_size = self.worker.tokens_per_expert.to_vec().len() * std::mem::size_of::<u32>();
        let source_dispatch_offset_size =
            self.worker.source_dispatch_offset.to_vec().len() * std::mem::size_of::<u32>();
        let combine_send_offset_size =
            self.worker.combine_send_offset.to_vec().len() * std::mem::size_of::<u32>();
        let source_rank_size = self.worker.source_rank.to_vec().len() * std::mem::size_of::<u32>();
        let padded_index_size = self.worker.padded_index.to_vec().len() * std::mem::size_of::<u32>();
        let num_recv_tokens_size =
            self.worker.num_recv_tokens.to_vec().len() * std::mem::size_of::<u32>();

        let overlaps = |a_ptr: usize, a_size: usize, b_ptr: usize, b_size: usize| {
            let a_end = a_ptr.saturating_add(a_size);
            let b_end = b_ptr.saturating_add(b_size);
            a_ptr < b_end && b_ptr < a_end
        };

        println!(
            "dispatch memory ranges {} rank={} expert_offsets=[0x{:x}, 0x{:x}) token_offset=[0x{:x}, 0x{:x}) send_buffer=0x{:x} recv_buffer=0x{:x} tokens_per_expert=[0x{:x}, 0x{:x}) source_dispatch_offset=[0x{:x}, 0x{:x}) combine_send_offset=[0x{:x}, 0x{:x}) source_rank=[0x{:x}, 0x{:x}) padded_index=[0x{:x}, 0x{:x}) num_recv_tokens=[0x{:x}, 0x{:x}) overlap_expert_tokens={} overlap_expert_source_dispatch={} overlap_expert_combine_send={} overlap_expert_source_rank={} overlap_expert_padded={} overlap_token_tokens={} overlap_token_source_dispatch={} overlap_token_combine_send={} overlap_token_source_rank={} overlap_token_padded={}",
            label,
            self.rank,
            expert_offsets_ptr,
            expert_offsets_ptr.saturating_add(expert_offsets_size),
            token_offset_ptr,
            token_offset_ptr.saturating_add(token_offset_size),
            send_buffer_ptr,
            recv_buffer_ptr,
            tokens_per_expert_ptr,
            tokens_per_expert_ptr.saturating_add(tokens_per_expert_size),
            source_dispatch_offset_ptr,
            source_dispatch_offset_ptr.saturating_add(source_dispatch_offset_size),
            combine_send_offset_ptr,
            combine_send_offset_ptr.saturating_add(combine_send_offset_size),
            source_rank_ptr,
            source_rank_ptr.saturating_add(source_rank_size),
            padded_index_ptr,
            padded_index_ptr.saturating_add(padded_index_size),
            num_recv_tokens_ptr,
            num_recv_tokens_ptr.saturating_add(num_recv_tokens_size),
            overlaps(
                expert_offsets_ptr,
                expert_offsets_size,
                tokens_per_expert_ptr,
                tokens_per_expert_size,
            ),
            overlaps(
                expert_offsets_ptr,
                expert_offsets_size,
                source_dispatch_offset_ptr,
                source_dispatch_offset_size,
            ),
            overlaps(
                expert_offsets_ptr,
                expert_offsets_size,
                combine_send_offset_ptr,
                combine_send_offset_size,
            ),
            overlaps(
                expert_offsets_ptr,
                expert_offsets_size,
                source_rank_ptr,
                source_rank_size,
            ),
            overlaps(
                expert_offsets_ptr,
                expert_offsets_size,
                padded_index_ptr,
                padded_index_size,
            ),
            overlaps(
                token_offset_ptr,
                token_offset_size,
                tokens_per_expert_ptr,
                tokens_per_expert_size,
            ),
            overlaps(
                token_offset_ptr,
                token_offset_size,
                source_dispatch_offset_ptr,
                source_dispatch_offset_size,
            ),
            overlaps(
                token_offset_ptr,
                token_offset_size,
                combine_send_offset_ptr,
                combine_send_offset_size,
            ),
            overlaps(
                token_offset_ptr,
                token_offset_size,
                source_rank_ptr,
                source_rank_size,
            ),
            overlaps(
                token_offset_ptr,
                token_offset_size,
                padded_index_ptr,
                padded_index_size,
            ),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_dim: usize,
        hidden_dim_scale: usize,
        in_elemsize: usize,
        out_elemsize: usize,
        out_dtype: ScalarType,
        scale_elemsize: usize,
        max_num_tokens: usize,
        max_recv_tokens: usize,
        max_private_tokens: usize,
        num_experts: usize,
        expert_padding: usize,
        num_experts_per_token: usize,
        rank: usize,
        dp_size: usize,
        node_size: usize,
        world_size: usize,
        num_routed_ptr: *mut u32,
        num_routed_mr: MemoryRegionHandle,
        send_buffer_ptr: *mut c_void,
        send_buffer_mr: MemoryRegionHandle,
        recv_buffer_ptr: *mut c_void,
        recv_buffer_mr: MemoryRegionHandle,
        sync_ptrs: Vec<u64>,
        send_ptrs: Vec<u64>,
        recv_ptrs: Vec<u64>,
        device: u8,
        imm_base: u32,
        rank_handles: Vec<AllToAllRankHandle>,
        transfer_engine: Arc<TransferEngine>,
        worker_cpu: Option<u16>,
    ) -> Result<Self> {
        // Start the all-to-all worker thread.
        /*
        for (i, peer) in rank_handles.iter().enumerate() {
            println!("Rank#{} Peer#{}: {}", rank, i, peer.address);
        }
        let output: String = rank_handles.iter()
            .enumerate()
            .map(|(i, peer)| format!("Rank#{} Peer#{}: {}", rank, i, peer.address))
            .collect::<Vec<String>>()
            .join("---");  // Join with separators
        println!("{}", output);
        */
        let worker: Arc<WorkerState> = Arc::new(WorkerState::new(
            hidden_dim,
            hidden_dim_scale,
            in_elemsize,
            out_elemsize,
            scale_elemsize,
            max_num_tokens,
            max_recv_tokens,
            max_private_tokens,
            num_experts,
            expert_padding,
            num_experts_per_token,
            rank,
            dp_size,
            node_size,
            world_size,
            num_routed_ptr,
            num_routed_mr,
            send_buffer_ptr,
            send_buffer_mr,
            recv_buffer_ptr,
            recv_buffer_mr,
            device,
            imm_base,
            rank_handles,
            transfer_engine,
        )?);

        // Create the worker thread.
        let (init_tx, init_rx) = oneshot::channel();


        let thread = {
            let thread_worker = worker.clone();
            Some(
                std::thread::Builder::new()
                    .name("p2p_all_to_all Worker".to_string())
                    .spawn(move || {
                        // Pin to the desired CPU.
                        //tracing::info!("Running worker for cuda:{}", device);
                        println!("Running worker for cuda:{}", device);
                        if let Some(cpu) = worker_cpu {
                            if let Err(e) = pin_cpu(cpu.into()) {
                                println!("Failed to pin CPU {}: {:?}", cpu, e);
                            }
                            println!(
                                "Pinned worker for cuda:{} to CPU {}",
                                device,
                                cpu
                            );
                        }

                        // Block until the worker is fully initialized.
                        if init_tx.send(()).is_err() {
                            panic!("Failed to send initialization signal");
                        } else {
                            println!("Initialized worker for cuda:{}", device);
                        }

                        // Main loop.
                        thread_worker.main_loop();
                        println!("Stopping worker for cuda:{}", device);
                    })
                    .expect("Failed to spawn p2p_all_to_all Worker thread"),
            )
        };
        init_rx.recv()?;

        let workspace = DeviceWorkspace::new(
            num_experts,
            max_num_tokens,
            num_experts_per_token,
            &sync_ptrs,
            &send_ptrs,
            &recv_ptrs,
        )?;

        let num_blocks = cudaGetNumSMs(device)?;
        println!(
            "AllToAllContext initialized for cuda:{} with num_blocks={} max_num_tokens={} num_experts={} num_experts_per_token={} max_private_tokens={}",
            device, num_blocks, max_num_tokens, num_experts, num_experts_per_token, max_private_tokens
        );

        // Build the context.
        Ok(Self {
            hidden_dim,
            hidden_dim_scale,
            in_elemsize,
            out_elemsize,
            out_dtype,
            scale_elemsize,
            num_experts,
            max_num_tokens,
            num_experts_per_token,
            max_private_tokens,
            expert_padding,
            rank,
            dp_size,
            node_size,
            world_size,
            device,
            workspace,
            worker,
            thread,
            num_blocks,
        })
    }

    /// Reset all ImmCounters/GdrCounters to 0 to prevent stale cycle-N CQEs
    /// from poisoning cycle-N+1 counters. Must only be called when the worker
    /// is idle (after wait_ready()) and all EFA operations have completed.
    pub fn reset_counters(&self) {
        self.worker.reset_counters();
    }

    /// Spin-wait until the worker thread has fully completed the current step
    /// (i.e. `tx_ready` is set). Call this before a cross-rank barrier between
    /// repetitions to ensure all in-flight EFA operations from this rank have
    /// drained before the next cycle starts.
    pub fn wait_ready(&self) {
        while !self.worker.tx_ready.is_set() {
            std::hint::spin_loop();
        }
    }

    pub fn destroy(&mut self) -> Result<()> {
        // Stop all work on the worker thread.
        println!("Stopping worker thread for cuda:{}", self.device);

        self.worker.stop();
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            return Err(anyhow!("Failed to join thread"));
        }
        Ok(())
    }

    pub fn debug_state(
        &self,
        max_token_offsets: Option<usize>,
        max_recv_entries: Option<usize>,
    ) -> Result<AllToAllDebugState> {
        let token_offset_len = self.max_num_tokens * self.num_experts_per_token;

        let token_offset = self.workspace.token_offset.to_vec::<u32>()?;
        let expert_offsets = self.workspace.expert_offsets.to_vec::<u32>()?;
        let combine_send_offset = self.worker.combine_send_offset.to_vec();
        let source_dispatch_offset = self.worker.source_dispatch_offset.to_vec();
        let source_rank = self.worker.source_rank.to_vec();
        let padded_index = self.worker.padded_index.to_vec();
        let num_recv_tokens = self.worker.num_recv_tokens.to_vec();
        let recv_len = combine_send_offset.len();
        let tokens_per_expert = self.worker.tokens_per_expert.to_vec();
        let sum_tokens_per_expert = tokens_per_expert.iter().copied().sum::<u32>();
        let num_recv_tokens_main = num_recv_tokens.first().copied().unwrap_or_default();
        let num_recv_efa_tokens = num_recv_tokens.get(1).copied().unwrap_or_default();
        let total_padded_tokens = tokens_per_expert
            .iter()
            .map(|&count| {
                let count = count as usize;
                count.div_ceil(self.expert_padding) * self.expert_padding
            })
            .sum::<usize>() as u32;
        let recv_entries = max_recv_entries.unwrap_or(recv_len).min(recv_len);
        let max_padded_index = padded_index.iter().take(recv_entries).copied().max();
        let padded_index_out_of_bounds = padded_index
            .iter()
            .take(recv_entries)
            .filter(|&&index| index >= total_padded_tokens)
            .count();

        Ok(AllToAllDebugState {
            tokens_per_expert,
            token_offset: token_offset[..max_token_offsets
                .unwrap_or(token_offset_len)
                .min(token_offset_len)]
                .to_vec(),
            expert_offsets,
            combine_send_offset: combine_send_offset
                [..max_recv_entries.unwrap_or(recv_len).min(recv_len)]
                .to_vec(),
            source_dispatch_offset: source_dispatch_offset
                [..max_recv_entries.unwrap_or(recv_len).min(recv_len)]
                .to_vec(),
            source_rank: source_rank[..max_recv_entries.unwrap_or(recv_len).min(recv_len)]
                .to_vec(),
            padded_index: padded_index[..max_recv_entries.unwrap_or(recv_len).min(recv_len)]
                .to_vec(),
            num_recv_tokens,
            sum_tokens_per_expert,
            num_recv_tokens_main,
            num_recv_efa_tokens,
            total_padded_tokens,
            max_padded_index,
            padded_index_out_of_bounds,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn dispatch_send(
        &mut self,
        num_tokens: usize,
        x_ptr: *const c_void,
        x_stride: usize,
        x_scale_ptr: *const c_void,
        x_scale_stride_elem: usize,
        x_scale_stride_token: usize,
        indices: *const i32,
        indices_stride: usize,
        weights: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        stream: u64,
    ) -> Result<()> {
        if num_tokens > self.max_num_tokens {
            return Err(anyhow!("Number of tokens exceeds maximum allowed"));
        }


        cuda_check!(a2a_kernels::a2a_dispatch_send(
            self.num_blocks,
            self.hidden_dim,
            self.hidden_dim_scale,
            self.num_experts,
            self.num_experts_per_token,
            self.max_private_tokens,
            self.rank,
            self.dp_size,
            self.node_size,
            self.world_size,
            num_tokens,
            bound_m_ptr,
            x_ptr as *const u8,
            self.in_elemsize,
            x_stride,
            x_scale_ptr as *const u8,
            self.scale_elemsize,
            x_scale_stride_elem,
            x_scale_stride_token,
            indices,
            indices_stride,
            weights,
            weights_stride,
            self.workspace.token_offset.get_mut_ptr(),
            self.worker.buffers.num_routed_ptr,
            self.workspace.expert_offsets.get_mut_ptr(),
            self.worker.dispatch_route_done.get_device_ptr(),
            self.worker.dispatch_send_done.get_device_ptr(),
            self.worker.tx_ready.get_device_ptr(),
            self.worker.buffers.send_buffer_ptr as *mut u8,
            self.workspace.grid_counter.get_mut_ptr(),
            self.workspace.sync_counter.get_mut_ptr(),
            self.workspace.get_sync_ptr(),
            self.workspace.get_recv_ptr() as *mut *mut u8,
            stream,
        ))?;

        Self::debug_sync_dispatch_stream(stream)?;

        if Self::debug_dispatch_offsets_enabled() {
            let expert_offsets = self.workspace.expert_offsets.to_vec::<u32>()?;
            let token_offset = self.workspace.token_offset.to_vec::<u32>()?;
            println!(
                "dispatch_send debug after_stream_sync rank={} expert_offsets={:?} token_offset_prefix={:?}",
                self.rank,
                &expert_offsets[..expert_offsets.len().min(16)],
                &token_offset[..token_offset.len().min(16)],
            );
        }

        if self.worker.failed() {
            return Err(anyhow!("fabric-lib transfer error"));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn dispatch_recv(
        &mut self,
        out_num_tokens_ptr: *mut i32,
        out_x_ptr: *mut c_void,
        out_x_stride: usize,
        out_prob_ptr: *mut f32,
        out_x_scale_ptr: *mut c_void,
        out_x_scale_stride_elem: usize,
        out_x_scale_stride_token: usize,
        stream: u64,
    ) -> Result<()> {
        self.debug_print_dispatch_memory_ranges("before_dispatch_recv");

        cuda_check!(a2a_kernels::a2a_dispatch_recv(
            self.num_blocks,
            self.hidden_dim,
            self.hidden_dim_scale,
            self.in_elemsize,
            self.scale_elemsize,
            self.num_experts,
            self.rank,
            self.node_size,
            self.world_size,
            out_num_tokens_ptr,
            out_x_ptr as *mut u8,
            out_x_stride,
            out_prob_ptr,
            out_x_scale_ptr as *mut u8,
            out_x_scale_stride_elem,
            out_x_scale_stride_token,
            self.worker.tokens_per_expert.get_device_ptr(),
            self.worker.buffers.send_buffer_ptr as *mut u8,
            self.worker.buffers.recv_buffer_ptr as *mut u8,
            self.worker.source_rank.get_device_ptr(),
            self.worker.source_dispatch_offset.get_device_ptr(),
            self.worker.padded_index.get_device_ptr(),
            self.worker.buffers.num_routed_ptr,
            self.worker.num_recv_tokens.get_device_ptr(),
            self.worker.num_recv_tokens_flag.get_device_ptr(),
            self.worker.dispatch_recv_flag.get_device_ptr(),
            self.worker.dispatch_recv_done.get_device_ptr(),
            self.workspace.grid_counter.get_mut_ptr(),
            self.workspace.sync_counter.get_mut_ptr(),
            self.workspace.get_sync_ptr(),
            self.workspace.get_send_ptr() as *mut *mut u8,
            stream,
        ))?;



        if self.worker.failed() {
            return Err(anyhow!("fabric-lib transfer error"));
        }

        self.debug_print_dispatch_memory_ranges("after_dispatch_recv");
        if Self::debug_dispatch_offsets_enabled() {
            let expert_offsets = self.workspace.expert_offsets.to_vec::<u32>()?;
            let token_offset = self.workspace.token_offset.to_vec::<u32>()?;
            println!(
                "dispatch_recv debug rank={} expert_offsets={:?} token_offset_prefix={:?}",
                self.rank,
                &expert_offsets[..expert_offsets.len().min(16)],
                &token_offset[..token_offset.len().min(16)],
            );
        }

        Ok(())
    }

    #[allow(
        unused_variables,
        clippy::too_many_arguments,
        clippy::not_unsafe_ptr_arg_deref
    )]
    pub fn combine_send(
        &mut self,
        expert_x_ptr: *const c_void,
        expert_x_stride: usize,
        stream: u64,
    ) -> Result<()> {
        cuda_check!(a2a_kernels::a2a_combine_send(
            self.num_blocks,
            self.hidden_dim,
            self.out_elemsize,
            self.rank,
            self.node_size,
            self.dp_size,
            expert_x_ptr as *const u8,
            expert_x_stride,
            self.worker.tx_ready.get_device_ptr(),
            self.worker.buffers.send_buffer_ptr as *mut u8,
            self.worker.buffers.recv_buffer_ptr as *mut u8,
            self.worker.source_rank.get_device_ptr(),
            self.worker.combine_send_offset.get_device_ptr(),
            self.worker.padded_index.get_device_ptr(),
            self.worker.num_recv_tokens.get_device_ptr(),
            self.worker.combine_send_done.get_device_ptr(),
            self.workspace.token_counter.get_mut_ptr(),
            self.workspace.sync_counter.get_mut_ptr(),
            self.workspace.get_sync_ptr(),
            self.workspace.get_recv_ptr() as *mut *mut u8,
            stream,
        ))?;

        if self.worker.failed() {
            return Err(anyhow!("fabric-lib transfer error"));
        }

        Ok(())
    }

    #[allow(
        unused_variables,
        clippy::too_many_arguments,
        clippy::not_unsafe_ptr_arg_deref
    )]
    pub fn combine_recv(
        &mut self,
        num_tokens: usize,
        num_recv_tokens: usize,
        expert_y_dtype: ScalarType,
        out_tokens_ptr: *mut c_void,
        out_tokens_stride: usize,
        indices_ptr: *const i32,
        indices_stride: usize,
        weights_ptr: *const f32,
        weights_stride: usize,
        bound_m_ptr: *const i32,
        accumulate: bool,
        stream: u64,
    ) -> Result<()> {
        cuda_check!(a2a_kernels::a2a_combine_recv(
            self.num_blocks,
            self.hidden_dim,
            self.out_elemsize,
            expert_y_dtype,
            self.out_dtype,
            self.num_experts,
            self.num_experts_per_token,
            self.rank,
            self.node_size,
            self.world_size,
            num_tokens,
            bound_m_ptr,
            indices_ptr,
            indices_stride,
            weights_ptr,
            weights_stride,
            out_tokens_ptr as *mut u8,
            out_tokens_stride,
            accumulate,
            self.worker.buffers.recv_buffer_ptr as *mut u8,
            self.workspace.token_offset.get_mut_ptr(),
            self.workspace.expert_offsets.get_mut_ptr(),
            self.worker.combine_recv_flag.get_device_ptr(),
            self.worker.combine_recv_done.get_device_ptr(),
            self.workspace.sync_counter.get_mut_ptr(),
            self.workspace.get_sync_ptr(),
            stream,
        ))?;

        if self.worker.failed() {
            return Err(anyhow!("fabric-lib transfer error"));
        }

        Ok(())
    }
}

impl Drop for AllToAllContext {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}
