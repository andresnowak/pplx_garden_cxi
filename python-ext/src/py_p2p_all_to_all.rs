use std::{
    ffi::c_void,
    ptr::{null, null_mut},
};

use p2p_all_to_all::{AllToAllContext, AllToAllRankHandle};
use pyo3::{
    Bound, PyResult, exceptions::PyRuntimeError, pyclass, pymethods, types::PyDict,
    types::PyDictMethods, types::PyModule, types::PyModuleMethods,
};
use torch_lib::ScalarType;

use crate::py_fabric_lib::{
    PyDomainAddress, PyMemoryRegionDescriptor, PyMemoryRegionHandle, PyTransferEngine,
};

#[pyclass(name = "AllToAllContext", module = "pplx_garden._rust")]
pub(crate) struct PyAllToAllContext {
    ctx: AllToAllContext,
}

#[pymethods]
impl PyAllToAllContext {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn create(
        hidden_dim: usize,
        hidden_dim_scale: Option<usize>,
        in_elemsize: usize,
        out_elemsize: usize,
        out_dtype: ScalarType,
        scale_elemsize: Option<usize>,
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
        num_routed_ptr: u64,
        num_routed_mr: PyMemoryRegionHandle,
        send_buffer_ptr: u64,
        send_buffer_mr: PyMemoryRegionHandle,
        recv_buffer_ptr: u64,
        recv_buffer_mr: PyMemoryRegionHandle,
        sync_ptrs: Vec<u64>,
        send_ptrs: Vec<u64>,
        recv_ptrs: Vec<u64>,
        device: u8,
        imm_base: u32,
        ranks: Vec<(
            PyDomainAddress,
            PyMemoryRegionDescriptor,
            PyMemoryRegionDescriptor,
        )>,
        transfer_engine: &PyTransferEngine,
        worker_cpu: Option<u16>,
    ) -> PyResult<Self> {
        let rank_handles = ranks
            .into_iter()
            .map(|data| AllToAllRankHandle::new(data.0.0, data.1.0, data.2.0))
            .collect();

        let ctx = AllToAllContext::new(
            hidden_dim,
            hidden_dim_scale.unwrap_or(0),
            in_elemsize,
            out_elemsize,
            out_dtype,
            scale_elemsize.unwrap_or(0),
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
            num_routed_ptr as *mut u32,
            num_routed_mr.0,
            send_buffer_ptr as *mut c_void,
            send_buffer_mr.0,
            recv_buffer_ptr as *mut c_void,
            recv_buffer_mr.0,
            sync_ptrs,
            send_ptrs,
            recv_ptrs,
            device,
            imm_base,
            rank_handles,
            transfer_engine.get_fabric_engine(),
            worker_cpu,
        )?;
        Ok(Self { ctx })
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_send(
        &mut self,
        num_tokens: usize,
        x_ptr: u64,
        x_stride: usize,
        x_scale_ptr: Option<u64>,
        x_scale_stride_elem: Option<usize>,
        x_scale_stride_token: Option<usize>,
        indices_ptr: u64,
        indices_stride: usize,
        weights_ptr: u64,
        weights_stride: usize,
        bound_m_ptr: Option<u64>,
        stream: u64,
    ) -> PyResult<()> {
        self.ctx
            .dispatch_send(
                num_tokens,
                x_ptr as *const c_void,
                x_stride,
                x_scale_ptr.map(|ptr| ptr as *const c_void).unwrap_or(null()),
                x_scale_stride_elem.unwrap_or(0),
                x_scale_stride_token.unwrap_or(0),
                indices_ptr as *const i32,
                indices_stride,
                weights_ptr as *const f32,
                weights_stride,
                bound_m_ptr.map(|ptr| ptr as *const i32).unwrap_or(null()),
                stream,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_recv(
        &mut self,
        out_num_tokens_ptr: u64,
        out_x_ptr: u64,
        out_x_stride: usize,
        out_prob_ptr: Option<u64>,
        out_x_scale_ptr: Option<u64>,
        out_x_scale_stride_elem: Option<usize>,
        out_x_scale_stride_token: Option<usize>,
        stream: u64,
    ) -> PyResult<()> {
        self.ctx
            .dispatch_recv(
                out_num_tokens_ptr as *mut i32,
                out_x_ptr as *mut c_void,
                out_x_stride,
                out_prob_ptr.map(|ptr| ptr as *mut f32).unwrap_or(null_mut()),
                out_x_scale_ptr.map(|ptr| ptr as *mut c_void).unwrap_or(null_mut()),
                out_x_scale_stride_elem.unwrap_or(0),
                out_x_scale_stride_token.unwrap_or(0),
                stream,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn combine_send(
        &mut self,
        expert_x_ptr: u64,
        expert_x_stride: usize,
        stream: u64,
    ) -> PyResult<()> {
        self.ctx
            .combine_send(expert_x_ptr as *const c_void, expert_x_stride, stream)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn combine_recv(
        &mut self,
        num_tokens: usize,
        num_recv_tokens: usize,
        expert_y_dtype: ScalarType,
        out_tokens_ptr: u64,
        out_tokens_stride: usize,
        indices_ptr: u64,
        indices_stride: usize,
        weights_ptr: u64,
        weights_stride: usize,
        bound_m_ptr: Option<u64>,
        accumulate: bool,
        stream: u64,
    ) -> PyResult<()> {
        self.ctx
            .combine_recv(
                num_tokens,
                num_recv_tokens,
                expert_y_dtype,
                out_tokens_ptr as *mut c_void,
                out_tokens_stride,
                indices_ptr as *const i32,
                indices_stride,
                weights_ptr as *const f32,
                weights_stride,
                bound_m_ptr.map(|ptr| ptr as *const i32).unwrap_or(null()),
                accumulate,
                stream,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (max_token_offsets=None, max_recv_entries=None))]
    fn debug_state<'py>(
        &self,
        py: pyo3::Python<'py>,
        max_token_offsets: Option<usize>,
        max_recv_entries: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let state = self
            .ctx
            .debug_state(max_token_offsets, max_recv_entries)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("tokens_per_expert", state.tokens_per_expert)?;
        dict.set_item("token_offset", state.token_offset)?;
        dict.set_item("expert_offsets", state.expert_offsets)?;
        dict.set_item("combine_send_offset", state.combine_send_offset)?;
        dict.set_item("source_dispatch_offset", state.source_dispatch_offset)?;
        dict.set_item("source_rank", state.source_rank)?;
        dict.set_item("padded_index", state.padded_index)?;
        dict.set_item("num_recv_tokens", state.num_recv_tokens)?;
        dict.set_item("sum_tokens_per_expert", state.sum_tokens_per_expert)?;
        dict.set_item("num_recv_tokens_main", state.num_recv_tokens_main)?;
        dict.set_item("num_recv_efa_tokens", state.num_recv_efa_tokens)?;
        dict.set_item("total_padded_tokens", state.total_padded_tokens)?;
        dict.set_item("max_padded_index", state.max_padded_index)?;
        dict.set_item(
            "padded_index_out_of_bounds",
            state.padded_index_out_of_bounds,
        )?;
        Ok(dict)
    }
}

pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAllToAllContext>()?;
    Ok(())
}
