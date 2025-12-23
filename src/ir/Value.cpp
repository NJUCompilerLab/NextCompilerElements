#include "ir/Value.h"
#include "ir/Operation.h"
#include "ir/Region.h"

namespace nc {
namespace ir {

void Value::replaceAllUsesWith(ValuePtr newValue) {
    // Copy uses_ since we'll be modifying it during iteration
    auto usesToUpdate = uses_;
    for (Operation* op : usesToUpdate) {
        op->replaceInput(shared_from_this(), newValue);
    }
}

} // namespace ir
} // namespace nc
