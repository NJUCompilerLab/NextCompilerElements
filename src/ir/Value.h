#pragma once

#include "ir/Type.h"
#include <string>
#include <memory>
#include <vector>
#include <algorithm>

namespace nc {
namespace ir {

class Operation;
class Value;
using ValuePtr = std::shared_ptr<Value>;

/// SSA Value - represents a named value with a type
/// In SSA form, each value is defined exactly once
class Value : public std::enable_shared_from_this<Value> {
public:
    Value(std::string name, Type type)
        : name_(std::move(name)), type_(std::move(type)), definingOp_(nullptr) {}

    /// SSA name (e.g., "%conv1", "%input")
    const std::string& name() const { return name_; }
    void setName(std::string name) { name_ = std::move(name); }

    /// Type of this value
    const Type& type() const { return type_; }
    Type& type() { return type_; }
    void setType(Type type) { type_ = std::move(type); }

    /// The operation that defines this value (nullptr for block arguments)
    Operation* definingOp() const { return definingOp_; }
    void setDefiningOp(Operation* op) { definingOp_ = op; }

    /// Check if this is a block argument (not defined by an operation)
    bool isBlockArg() const { return definingOp_ == nullptr; }

    // ========== Use-Def Chain ==========

    /// Get all operations that use this value
    const std::vector<Operation*>& uses() const { return uses_; }

    /// Add a use (called when Operation adds this value as input)
    void addUse(Operation* op) {
        if (op && std::find(uses_.begin(), uses_.end(), op) == uses_.end()) {
            uses_.push_back(op);
        }
    }

    /// Remove a use (called when Operation removes this value from inputs)
    void removeUse(Operation* op) {
        uses_.erase(std::remove(uses_.begin(), uses_.end(), op), uses_.end());
    }

    /// Check if this value has any uses
    bool hasUses() const { return !uses_.empty(); }

    /// Number of uses
    size_t numUses() const { return uses_.size(); }

    /// Replace all uses of this value with another value
    /// Implementation in Value.cpp to avoid circular dependency
    void replaceAllUsesWith(ValuePtr newValue);

private:
    std::string name_;
    Type type_;
    Operation* definingOp_;
    std::vector<Operation*> uses_;
};

} // namespace ir
} // namespace nc
