#pragma once

#include "ir/Module.h"
#include "ir/Operation.h"
#include "ir/Block.h"
#include "ir/Region.h"
#include "ir/Value.h"
#include "ir/Type.h"
#include <string>
#include <memory>

namespace nc {
namespace ir {

/// IRBuilder - convenience API for building IR
class IRBuilder {
public:
    explicit IRBuilder(Block* block = nullptr) : insertBlock_(block), valueCounter_(0) {}

    /// Set insertion point
    void setInsertionPoint(Block* block) { insertBlock_ = block; }
    Block* getInsertionBlock() const { return insertBlock_; }

    /// Generate a unique SSA name
    std::string genName(const std::string& prefix = "v") {
        return "%" + prefix + std::to_string(valueCounter_++);
    }

    // ========== Create Operations ==========

    /// Create a generic operation
    ValuePtr createOp(const std::string& opType,
                      const std::vector<ValuePtr>& inputs,
                      Type resultType,
                      const Attributes& attrs = {}) {
        if (!insertBlock_) {
            throw std::runtime_error("IRBuilder: no insertion point set");
        }
        auto name = genName();
        auto op = std::make_unique<Operation>(name, opType);
        op->setInputs(inputs);
        op->setOutputTypes({resultType});
        for (const auto& [k, v] : attrs) {
            op->setAttr(k, v);
        }

        auto result = std::make_shared<Value>(name, resultType);
        op->addResult(result);

        insertBlock_->addOp(std::move(op));
        return result;
    }

    // ========== Shape Inference Helpers ==========

    /// Infer output type for elementwise binary ops with broadcasting
    /// Returns the broadcasted type (larger shape wins for each dim)
    static Type inferBroadcastType(const Type& lhs, const Type& rhs) {
        const auto& lhsDims = lhs.shape().dims();
        const auto& rhsDims = rhs.shape().dims();

        // Start from the right, pad shorter with 1s
        size_t maxRank = std::max(lhsDims.size(), rhsDims.size());
        std::vector<int64_t> resultDims(maxRank);

        for (size_t i = 0; i < maxRank; ++i) {
            int64_t lhsDim = (i < lhsDims.size()) ? lhsDims[lhsDims.size() - 1 - i] : 1;
            int64_t rhsDim = (i < rhsDims.size()) ? rhsDims[rhsDims.size() - 1 - i] : 1;
            resultDims[maxRank - 1 - i] = std::max(lhsDim, rhsDim);
        }

        return Type(lhs.dtype(), Shape(resultDims));
    }

    /// Infer output type for MatMul: [..., M, K] @ [..., K, N] -> [..., M, N]
    static Type inferMatMulType(const Type& lhs, const Type& rhs) {
        const auto& lhsDims = lhs.shape().dims();
        const auto& rhsDims = rhs.shape().dims();

        if (lhsDims.size() < 2 || rhsDims.size() < 2) {
            throw std::runtime_error("MatMul requires at least 2D tensors");
        }

        std::vector<int64_t> resultDims;
        // Batch dimensions (broadcast)
        size_t lhsBatch = lhsDims.size() - 2;
        size_t rhsBatch = rhsDims.size() - 2;
        size_t maxBatch = std::max(lhsBatch, rhsBatch);
        for (size_t i = 0; i < maxBatch; ++i) {
            int64_t l = (i < lhsBatch) ? lhsDims[lhsBatch - 1 - i] : 1;
            int64_t r = (i < rhsBatch) ? rhsDims[rhsBatch - 1 - i] : 1;
            resultDims.insert(resultDims.begin(), std::max(l, r));
        }

        // Matrix dimensions: [M, K] @ [K, N] -> [M, N]
        resultDims.push_back(lhsDims[lhsDims.size() - 2]);  // M
        resultDims.push_back(rhsDims[rhsDims.size() - 1]);  // N

        return Type(lhs.dtype(), Shape(resultDims));
    }

    // ========== Common Operations ==========

    /// MatMul: C = A @ B (with automatic shape inference)
    ValuePtr createMatMul(ValuePtr lhs, ValuePtr rhs) {
        return createOp("MatMul", {lhs, rhs}, inferMatMulType(lhs->type(), rhs->type()));
    }

    /// MatMul with explicit result type (backward compatible)
    ValuePtr createMatMul(ValuePtr lhs, ValuePtr rhs, Type resultType) {
        return createOp("MatMul", {lhs, rhs}, resultType);
    }

    /// Add: C = A + B (with automatic shape inference)
    ValuePtr createAdd(ValuePtr lhs, ValuePtr rhs) {
        return createOp("Add", {lhs, rhs}, inferBroadcastType(lhs->type(), rhs->type()));
    }

    /// Add with explicit result type (backward compatible)
    ValuePtr createAdd(ValuePtr lhs, ValuePtr rhs, Type resultType) {
        return createOp("Add", {lhs, rhs}, resultType);
    }

    /// ReLU: y = max(0, x)
    ValuePtr createReLU(ValuePtr input) {
        return createOp("ReLU", {input}, input->type());
    }

    /// GELU activation
    ValuePtr createGELU(ValuePtr input) {
        return createOp("GELU", {input}, input->type());
    }

    /// LayerNorm
    ValuePtr createLayerNorm(ValuePtr input, ValuePtr weight, ValuePtr bias,
                             int64_t axis, double eps = 1e-5) {
        Attributes attrs;
        attrs["axis"] = axis;
        attrs["eps"] = eps;
        return createOp("LayerNorm", {input, weight, bias}, input->type(), attrs);
    }

    /// Softmax
    ValuePtr createSoftmax(ValuePtr input, int64_t axis) {
        Attributes attrs;
        attrs["axis"] = axis;
        return createOp("Softmax", {input}, input->type(), attrs);
    }

    /// Embedding lookup
    ValuePtr createEmbedding(ValuePtr indices, ValuePtr weight, Type resultType) {
        return createOp("Embedding", {indices, weight}, resultType);
    }

    /// Permute (transpose dimensions)
    ValuePtr createPermute(ValuePtr input, std::vector<int64_t> perm, Type resultType) {
        Attributes attrs;
        attrs["perm"] = std::move(perm);
        return createOp("Permute", {input}, resultType, attrs);
    }

    /// Reshape
    ValuePtr createReshape(ValuePtr input, std::vector<int64_t> shape, Type resultType) {
        Attributes attrs;
        attrs["shape"] = std::move(shape);
        return createOp("Reshape", {input}, resultType, attrs);
    }

    /// Mul: C = A * B (with automatic shape inference)
    ValuePtr createMul(ValuePtr lhs, ValuePtr rhs) {
        return createOp("Mul", {lhs, rhs}, inferBroadcastType(lhs->type(), rhs->type()));
    }

    /// Mul with explicit result type (backward compatible)
    ValuePtr createMul(ValuePtr lhs, ValuePtr rhs, Type resultType) {
        return createOp("Mul", {lhs, rhs}, resultType);
    }

    /// Div: C = A / B (with automatic shape inference)
    ValuePtr createDiv(ValuePtr lhs, ValuePtr rhs) {
        return createOp("Div", {lhs, rhs}, inferBroadcastType(lhs->type(), rhs->type()));
    }

    /// Div with explicit result type (backward compatible)
    ValuePtr createDiv(ValuePtr lhs, ValuePtr rhs, Type resultType) {
        return createOp("Div", {lhs, rhs}, resultType);
    }

    /// Where: result = cond ? x : y
    ValuePtr createWhere(ValuePtr cond, ValuePtr x, ValuePtr y) {
        return createOp("Where", {cond, x, y}, x->type());
    }

    /// Constant
    ValuePtr createConstant(Type type, AttrValue value) {
        auto name = genName("const");
        auto op = std::make_unique<Operation>(name, "Constant");
        op->setAttr("value", std::move(value));
        op->setAttr("dtype", dtypeToString(type.dtype()));
        op->setAttr("shape", type.shape().dims());
        op->setOutputTypes({type});

        auto result = std::make_shared<Value>(name, type);
        op->addResult(result);

        insertBlock_->addOp(std::move(op));
        return result;
    }

    // ========== Terminator ==========

    void createReturn(const std::vector<ValuePtr>& values) {
        auto op = std::make_unique<Operation>("", "Return");
        op->setInputs(values);
        insertBlock_->setTerminator(std::move(op));
    }

    /// Unconditional branch: br @dest(args)
    void createBr(Block* dest, const std::vector<ValuePtr>& args = {}) {
        auto op = std::make_unique<Operation>("", "Br");
        op->setAttr("dest", dest->label());
        op->setInputs(args);
        insertBlock_->setTerminator(std::move(op));
    }

    /// Conditional branch: cond_br %cond, @then_block, @else_block
    void createCondBr(ValuePtr cond, Block* thenBlock, Block* elseBlock) {
        auto op = std::make_unique<Operation>("", "CondBr");
        op->setInputs({cond});
        op->setAttr("then", thenBlock->label());
        op->setAttr("else", elseBlock->label());
        insertBlock_->setTerminator(std::move(op));
    }

private:
    Block* insertBlock_;
    int valueCounter_;
};

} // namespace ir
} // namespace nc
