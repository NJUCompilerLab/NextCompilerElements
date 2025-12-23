#include "ir/Operation.h"
#include "ir/Region.h"

namespace nc {
namespace ir {

// Destructor defined here where Region is complete
Operation::~Operation() = default;

Region* Operation::addRegion() {
    auto region = std::make_unique<Region>();
    region->setParentOp(this);
    regions_.push_back(std::move(region));
    return regions_.back().get();
}

} // namespace ir
} // namespace nc
