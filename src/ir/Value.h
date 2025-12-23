#pragma once

#include "ir/Type.h"
#include <string>
#include <memory>

namespace nc {
namespace ir {

class Operation;

/// SSA Value - represents a named value with a type
/// In SSA form, each value is defined exactly once
class Value {
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

private:
    std::string name_;
    Type type_;
    Operation* definingOp_;
};

/// Shared pointer for Value (values may be referenced by multiple operations)
using ValuePtr = std::shared_ptr<Value>;

} // namespace ir
} // namespace nc
