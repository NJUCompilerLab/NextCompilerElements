#pragma once

#include "ir/Block.h"
#include <vector>
#include <memory>

namespace nc {
namespace ir {

class Operation;

/// Region - a container of Blocks
/// Represents a scope for control flow
class Region {
public:
    Region() : parentOp_(nullptr) {}

    // ========== Blocks ==========

    const std::vector<BlockPtr>& blocks() const { return blocks_; }
    std::vector<BlockPtr>& blocks() { return blocks_; }

    Block* addBlock(std::string label = "@entry") {
        auto block = std::make_unique<Block>(std::move(label));
        block->setParentRegion(this);
        blocks_.push_back(std::move(block));
        return blocks_.back().get();
    }

    Block* entryBlock() const {
        return blocks_.empty() ? nullptr : blocks_[0].get();
    }

    size_t numBlocks() const { return blocks_.size(); }

    // ========== Parent ==========

    Operation* parentOp() const { return parentOp_; }
    void setParentOp(Operation* op) { parentOp_ = op; }

private:
    std::vector<BlockPtr> blocks_;
    Operation* parentOp_;
};

using RegionPtr = std::unique_ptr<Region>;

} // namespace ir
} // namespace nc
