#pragma once

#include "ir/Value.h"
#include "ir/Type.h"
#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <unordered_map>

namespace nc {
namespace ir {

class Region;
class Block;

/// Attribute value types
using AttrValue = std::variant<
    int64_t,                    // int
    double,                     // float
    bool,                       // bool
    std::string,                // string
    std::vector<int64_t>,       // int[]
    std::vector<double>         // float[]
>;

/// Attributes map
using Attributes = std::unordered_map<std::string, AttrValue>;

/// Operation - the core IR node
/// Everything is an Operation: Module, Function, Conv2D, ReLU, etc.
class Operation {
public:
    explicit Operation(std::string opType)
        : opType_(std::move(opType)), parentBlock_(nullptr) {}

    // Destructor must be defined in .cpp where Region is complete
    ~Operation();

    /// Get name: for symbol ops (Function/Module) returns sym_name attr,
    /// for regular ops returns first result's name
    std::string name() const {
        // Symbol operations store name in sym_name attribute
        if (hasAttr("sym_name")) {
            return "@" + getAttr<std::string>("sym_name");
        }
        // Regular ops: use first result's name
        if (!results_.empty()) {
            return results_[0]->name();
        }
        return "";
    }

    /// Operator type (e.g., "Conv2D", "Function", "Module")
    const std::string& opType() const { return opType_; }

    // ========== Inputs ==========

    /// Input values (SSA references)
    const std::vector<ValuePtr>& inputs() const { return inputs_; }

    void addInput(ValuePtr value) {
        value->addUse(this);
        inputs_.push_back(std::move(value));
    }

    void setInputs(std::vector<ValuePtr> inputs) {
        // Remove old uses
        for (auto& v : inputs_) {
            v->removeUse(this);
        }
        // Add new uses
        for (auto& v : inputs) {
            v->addUse(this);
        }
        inputs_ = std::move(inputs);
    }

    /// Replace an input value with a new value
    void replaceInput(ValuePtr oldValue, ValuePtr newValue) {
        for (auto& input : inputs_) {
            if (input == oldValue) {
                oldValue->removeUse(this);
                newValue->addUse(this);
                input = newValue;
            }
        }
    }

    // ========== Outputs ==========

    /// Output types (derived from results)
    std::vector<Type> outputTypes() const {
        std::vector<Type> types;
        types.reserve(results_.size());
        for (const auto& r : results_) {
            types.push_back(r->type());
        }
        return types;
    }

    /// Result values (created lazily or by builder)
    const std::vector<ValuePtr>& results() const { return results_; }
    void addResult(ValuePtr value) {
        value->setDefiningOp(this);
        results_.push_back(std::move(value));
    }
    void setResults(std::vector<ValuePtr> results) {
        for (auto& r : results) r->setDefiningOp(this);
        results_ = std::move(results);
    }

    /// Get the single result (convenience for ops with one output)
    ValuePtr result() const {
        return results_.empty() ? nullptr : results_[0];
    }

    // ========== Attributes ==========

    const Attributes& attrs() const { return attrs_; }
    Attributes& attrs() { return attrs_; }

    void setAttr(const std::string& key, AttrValue value) {
        attrs_[key] = std::move(value);
    }

    bool hasAttr(const std::string& key) const {
        return attrs_.find(key) != attrs_.end();
    }

    template<typename T>
    T getAttr(const std::string& key) const {
        return std::get<T>(attrs_.at(key));
    }

    template<typename T>
    T getAttrOr(const std::string& key, T defaultValue) const {
        auto it = attrs_.find(key);
        if (it == attrs_.end()) return defaultValue;
        return std::get<T>(it->second);
    }

    // ========== Regions (for nested structure) ==========

    const std::vector<std::unique_ptr<Region>>& regions() const { return regions_; }
    std::vector<std::unique_ptr<Region>>& regions() { return regions_; }

    Region* addRegion();
    size_t numRegions() const { return regions_.size(); }
    Region* getRegion(size_t i) const {
        return i < regions_.size() ? regions_[i].get() : nullptr;
    }

    // ========== Parent ==========

    Block* parentBlock() const { return parentBlock_; }
    void setParentBlock(Block* block) { parentBlock_ = block; }

    // ========== Utilities ==========

    /// Check if this is a terminator operation
    bool isTerminator() const {
        return opType_ == "Return" || opType_ == "Br" || opType_ == "CondBr";
    }

private:
    std::string opType_;
    std::vector<ValuePtr> inputs_;
    std::vector<ValuePtr> results_;
    Attributes attrs_;
    std::vector<std::unique_ptr<Region>> regions_;
    Block* parentBlock_;
};

using OperationPtr = std::unique_ptr<Operation>;

} // namespace ir
} // namespace nc
