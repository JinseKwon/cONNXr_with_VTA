//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__averagepool__1.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__averagepool__1(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);

    TRACE_TENSOR(2, true, i_X);

    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    // context_operator__ai_onnx__globalaveragepool__1 *op_ctx = NULL;
    // op_ctx = malloc(sizeof(context_operator__ai_onnx__globalaveragepool__1));
    // TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */

    o_Y->has_raw_data = 0;
    o_Y->data_type    = i_X->data_type;
    o_Y->n_dims       = i_X->n_dims;
    o_Y->dims         = ARRAYDUP(i_X->dims, i_X->n_dims);

    for (int i = 2; i < o_Y->n_dims; i++) {
      o_Y->dims[i] = 1;
    }

    /* MALLOC OUTPUT TENSORS HERE */

    mallocTensorData(o_Y);

    TRACE_TENSOR(2, true, o_Y);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    ctx->executer = resolve_operator__ai_onnx__averagepool__1(ctx);
    // ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    // return OP_ENOSYS;
    return OP_OK;
}