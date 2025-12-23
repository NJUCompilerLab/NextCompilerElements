#pragma once

#include "ir/Operation.h"
#include "ir/Region.h"
#include "ir/Block.h"
#include <memory>
#include <string>

namespace nc {
namespace ir {

/// Module - the top-level container for an IR program
/// A Module is itself an Operation with op_type="Module"
class Module {
public:
    Module(std::string name = "module")
        : op_(std::make_unique<Operation>("Module")) {
        // Store module name as sym_name attribute (strip @ prefix if present)
        if (!name.empty() && name[0] == '@') {
            name = name.substr(1);
        }
        op_->setAttr("sym_name", std::move(name));
        // Create the default region and entry block
        auto* region = op_->addRegion();
        region->addBlock("@entry");
    }

    /// Get the underlying Operation
    Operation* op() const { return op_.get(); }

    /// Get the module name
    std::string name() const { return op_->name(); }

    /// Get the main region
    Region* region() const { return op_->getRegion(0); }

    /// Get the entry block (where functions are defined)
    Block* entryBlock() const {
        auto* r = region();
        return r ? r->entryBlock() : nullptr;
    }

    /// Add a function to the module
    Operation* addFunction(const std::string& name) {
        auto funcOp = std::make_unique<Operation>("Function");
        // Store function name as sym_name attribute (without @ prefix)
        funcOp->setAttr("sym_name", name.substr(0, 1) == "@" ? name.substr(1) : name);
        // Function has its own region
        funcOp->addRegion()->addBlock("@entry");
        return entryBlock()->addOp(std::move(funcOp));
    }

private:
    OperationPtr op_;
};

using ModulePtr = std::unique_ptr<Module>;

} // namespace ir
} // namespace nc
