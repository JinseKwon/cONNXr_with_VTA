//this file was generated by ../../../../../../scripts/onnx_generator/OperatorTemplate.py
#include "operator__ai_onnx__elu__6.h"
#include "tracing.h"
#include "utils.h"

operator_status
prepare_operator__ai_onnx__elu__6(
    node_context *ctx
)
{
    TRACE_ENTRY(1);

    TRACE_NODE(2, true, ctx->onnx_node);

    /* UNCOMMENT AS NEEDED */

    Onnx__TensorProto *i_X = searchInputByName(ctx, 0);

    TRACE_TENSOR(2, true, i_X);

    Onnx__AttributeProto *a_alpha = searchAttributeNyName(ctx->onnx_node->n_attribute,ctx->onnx_node->attribute,"alpha");

    TRACE_ATTRIBUTE(2, a_alpha, a_alpha);

    Onnx__TensorProto *o_Y = searchOutputByName(ctx, 0);

    /* ALLOCATE AND INITIALIZE CONTEXT HERE IF NEEDED */

    float default_alpha = 1.0;

    context_operator__ai_onnx__elu__6 *op_ctx = NULL;
    op_ctx = (context_operator__ai_onnx__elu__6*)malloc(sizeof(context_operator__ai_onnx__elu__6));
    TRACE_FATAL(0 , !op_ctx, "could not allocate executer_context");

    op_ctx->alpha = a_alpha?a_alpha->f:default_alpha;

    TRACE_VAR(2, true, op_ctx->alpha, "%f");

    /* INITIALIZE OUTPUTS DATA_TYPE AND SHAPE HERE */

    o_Y->n_dims       = i_X->n_dims;
    o_Y->has_raw_data = 0;
    o_Y->data_type    = i_X->data_type;
    o_Y->dims         = ARRAYDUP(i_X->dims,i_X->n_dims);

    /* MALLOC OUTPUT TENSORS HERE */

    mallocTensorData(o_Y);

    TRACE_TENSOR(2, true, o_Y);

    /* CHOOSE EXECUTER AND CONTEXT HERE */
    /* YOU MAY USE THE GENERATED RESOLVER */

    ctx->executer = resolve_operator__ai_onnx__elu__6(ctx);
    ctx->executer_context = op_ctx;

    TRACE_EXIT(1);

    /* CHANGE RETURN CODE IF THIS PREPARER IS VALID */
    // return OP_ENOSYS;
    return OP_OK;
}