#include "Types.h"
#include <sstream>
#include <algorithm>

std::string toString(BaseType type) {
    switch (type) {
        case BaseType::UNKNOWN:    return "unknown";
        case BaseType::BOOL:       return "boolean";
        case BaseType::CHARACTER:  return "character";
        case BaseType::INTEGER:    return "integer";
        case BaseType::VECTOR:     return "vector";
        case BaseType::REAL:       return "real";
        case BaseType::TUPLE:      return "tuple";
        case BaseType::STRING:     return "string";
        case BaseType::STRUCT:     return "struct";
        case BaseType::ARRAY:      return "array";
        case BaseType::MATRIX:     return "matrix";
        case BaseType::UNRESOLVED: return "unresolved";
    }
    throw std::runtime_error("toString: FATAL: No string representation found for type");
}

std::string toString(const CompleteType& type) {
    std::ostringstream oss;
    oss << toString(type.baseType);

    if (std::find(std::begin(compositeTypes), std::end(compositeTypes), type.baseType) != std::end(compositeTypes)) {
        oss << "<";
        for (size_t i = 0; i < type.subTypes.size(); ++i) {
            oss << toString(type.subTypes[i]);
            if (i + 1 < type.subTypes.size()) {
                oss << ", ";
            }
        }
        oss << ">";
    }


    return oss.str();
}
/* TODO pt2
    - handle character array <-> character vector <-> string promotions
*/
BaseType promote(BaseType from, BaseType to)
{
    switch (from) {
        case BaseType::UNKNOWN:
            return BaseType::UNKNOWN;
            break;
        case BaseType::BOOL:
            switch(to) {
                case BaseType::BOOL:       return BaseType::BOOL;

                // Implicit promotions
                case BaseType::ARRAY:      return BaseType::ARRAY;
                case BaseType::VECTOR:     return BaseType::VECTOR; 
                case BaseType::MATRIX:     return BaseType::MATRIX;
            }
            break;
        case BaseType::CHARACTER:
            switch (to) {
                case BaseType::CHARACTER:  return BaseType::CHARACTER; 
            }
            break;
        case BaseType::INTEGER:
            switch (to) {
                case BaseType::INTEGER:    return BaseType::INTEGER; 
                case BaseType::REAL:       return BaseType::REAL;

                // Implicit promotions
                case BaseType::ARRAY:      return BaseType::ARRAY;
                case BaseType::VECTOR:     return BaseType::VECTOR; 
                case BaseType::MATRIX:     return BaseType::MATRIX;
            }
            break;
        case BaseType::VECTOR:
            switch (to) {
                case BaseType::VECTOR:     return BaseType::VECTOR;
            }
            break;
        case BaseType::REAL:
            switch (to) {
                case BaseType::REAL:       return BaseType::REAL;

                // Implicit promotions
                case BaseType::ARRAY:      return BaseType::ARRAY;
                case BaseType::VECTOR:     return BaseType::VECTOR; 
                case BaseType::MATRIX:     return BaseType::MATRIX;
            }
            break;
        case BaseType::TUPLE:
            switch (to) {
                case BaseType::TUPLE:      return BaseType::TUPLE;    // only valid if same len + internal types are convertible element-wise
            }
            break;
        case BaseType::STRING:
            switch (to) {
                case BaseType::STRING:     return BaseType::STRING;   // not the same as vector<char>. 
            }
            break;
        case BaseType::STRUCT:
            switch (to) {
                case BaseType::STRUCT:     return BaseType::STRUCT;    
            }
            break;
        case BaseType::ARRAY:
            switch (to) {
                case BaseType::ARRAY:      return BaseType::ARRAY;    // only valid if same len
            }
            break;
        case BaseType::MATRIX:
            switch (to) {
                case BaseType::MATRIX:     return BaseType::MATRIX;   // only valid if same len - special case with multiplication.
            }
            break;
    }
    return BaseType::UNKNOWN;
}

CompleteType promote(const CompleteType& from, const CompleteType& to) {
    validateSubtypes(from);
    validateSubtypes(to);

    /* TODO pt2
        Scalar to Composite
        - Implement checks on scalar -> composite only when subtypes align
        - Implement conversion from scalar -> composite where applicable

        Composite to Composite
        - Implement size checks (arr, vec, matrix)
        - Handle matmul edge case
        - Handle struct member-wise checks 
    */
    switch(from.baseType) {
        case BaseType::UNKNOWN:
            return BaseType::UNKNOWN;
            break;
        case BaseType::BOOL:
            switch(to.baseType) {
                case BaseType::BOOL:       return BaseType::BOOL;

                // Implicit promotions
                case BaseType::ARRAY:      return BaseType::ARRAY;
                case BaseType::VECTOR:     return BaseType::VECTOR; 
                case BaseType::MATRIX:     return BaseType::MATRIX;
            }
            break;
        case BaseType::CHARACTER:
            switch (to.baseType) {
                case BaseType::CHARACTER:  return BaseType::CHARACTER; 
            }
            break;
        case BaseType::INTEGER:
            switch (to.baseType) {
                case BaseType::INTEGER:    return BaseType::INTEGER; 
                case BaseType::REAL:       return BaseType::REAL;

                // Implicit promotions
                case BaseType::ARRAY:      return BaseType::ARRAY;
                case BaseType::VECTOR:     return BaseType::VECTOR; 
                case BaseType::MATRIX:     return BaseType::MATRIX;
            }
            break;
        case BaseType::REAL:
            switch (to.baseType) {
                case BaseType::REAL:       return BaseType::REAL;

                // Implicit promotions
                case BaseType::ARRAY:      return BaseType::ARRAY;
                case BaseType::VECTOR:     return BaseType::VECTOR; 
                case BaseType::MATRIX:     return BaseType::MATRIX;
            }
            break;
       case BaseType::TUPLE:
            switch (to.baseType) {
                case BaseType::TUPLE: {
                    // Check if tuple lengths match
                    if (from.subTypes.size() != to.subTypes.size()) {
                        return CompleteType(BaseType::UNKNOWN);
                    }
                    // Check if each element type can be promoted to the corresponding target type
                    CompleteType result(BaseType::TUPLE);
                    result.subTypes.reserve(from.subTypes.size());
                    for (size_t i = 0; i < from.subTypes.size(); ++i) {
                        CompleteType promotedElem = promote(from.subTypes[i], to.subTypes[i]);
                        if (promotedElem.baseType == BaseType::UNKNOWN) {
                            return CompleteType(BaseType::UNKNOWN);
                        }
                        result.subTypes.push_back(promotedElem);
                    }
                    return result;
                }
            }
            break;
        case BaseType::STRING:
            switch (to.baseType) {
                case BaseType::STRING:     return BaseType::STRING;   // not the same as vector<char>. 
            }
            break;
        case BaseType::STRUCT:
            switch (to.baseType) {
                case BaseType::STRUCT: {
                    if (from.subTypes.size() != to.subTypes.size()) return CompleteType(BaseType::UNKNOWN);

                    CompleteType result(BaseType::STRUCT);
                    result.subTypes.reserve(from.subTypes.size());
                    for (size_t i = 0; i < from.subTypes.size(); ++i) {
                        CompleteType promotedElem = promote(from.subTypes[i], to.subTypes[i]);
                        if (promotedElem.baseType == BaseType::UNKNOWN) return CompleteType(BaseType::UNKNOWN);

                        result.subTypes.push_back(promotedElem);
                    }
                    return result;
                }
            }
            break;
        case BaseType::ARRAY:
            if (from.subTypes.size() != 1) {
                throw std::runtime_error(
                    "promote(): from array with subtype len " +
                    std::to_string(from.subTypes.size()));
            }

            switch (to.baseType) {
                case BaseType::ARRAY: {
                    // TODO: Add len checks
                    if (to.subTypes.size() != 1) {
                        throw std::runtime_error(
                            "promote(): to array with subtype len " +
                            std::to_string(to.subTypes.size()));
                    }

                    CompleteType result(BaseType::ARRAY);
                    CompleteType subtypeResult = promote(from.subTypes[0], to.subTypes[0]);

                    result.subTypes.push_back(subtypeResult);

                    return result;
                }

                case BaseType::VECTOR: {

                    if (to.subTypes.size() != 1) {
                        throw std::runtime_error(
                            "promote(): to vector with subtype len " +
                            std::to_string(to.subTypes.size()));
                    }

                    CompleteType result(BaseType::VECTOR);
                    CompleteType subtypeResult = promote(from.subTypes[0], to.subTypes[0]);

                    result.subTypes.push_back(subtypeResult);

                    return result;
                }
            }
            break;
        case BaseType::VECTOR:
            if (from.subTypes.size() != 1) {
                throw std::runtime_error(
                    "promote(): from vector with subtype len " +
                    std::to_string(from.subTypes.size()));
            }

            switch (to.baseType) {
                case BaseType::VECTOR: {
                    if (to.subTypes.size() != 1) {
                        throw std::runtime_error(
                            "promote(): to vector with subtype len " +
                            std::to_string(to.subTypes.size()));
                    }

                    CompleteType result(BaseType::VECTOR);
                    CompleteType subtypeResult = promote(from.subTypes[0], to.subTypes[0]);

                    result.subTypes.push_back(subtypeResult);

                    return result;
                }
                case BaseType::ARRAY: {
                    if (to.subTypes.size() != 1) {
                        throw std::runtime_error(
                            "promote(): to array with subtype len " +
                            std::to_string(to.subTypes.size()));
                    }

                    CompleteType result(BaseType::ARRAY);
                    CompleteType subtypeResult = promote(from.subTypes[0], to.subTypes[0]);

                    result.subTypes.push_back(subtypeResult);

                    return result;
                }
            }
            break;
        case BaseType::MATRIX:
            switch (to.baseType) {
                case BaseType::MATRIX:     return BaseType::MATRIX;   // only valid if same len - special case with multiplication.
            }
            break;
    }
    return BaseType::UNKNOWN;
}

void validateSubtypes(CompleteType completeType) {
    // Flat type checking
    if (std::find(std::begin(flatTypes), std::end(flatTypes), completeType.baseType) != std::end(flatTypes)) {

        if (!completeType.subTypes.empty()) {
            throw std::runtime_error("Semantic Validation: Non-composite type of '" + toString(completeType.baseType) + "' cannot have subtypes.");
        }

    } else { // composite type validation
        if (completeType.baseType == BaseType::TUPLE && completeType.subTypes.size() < 2) {
            throw std::runtime_error("Semantic Validation: Tuple must have at least 2 subtypes.");
        }

        if (
            (completeType.baseType == BaseType::ARRAY || completeType.baseType == BaseType::VECTOR || completeType.baseType == BaseType::MATRIX) &&
            completeType.subTypes.size() != 1
    ) {
            throw std::runtime_error("Semantic Validation:" + toString(completeType.baseType) + " cannot have " + std::to_string(completeType.subTypes.size()) + " types.");
        }
    }
}

bool isScalarType(BaseType t) {
    for (auto ft : flatTypes) {
        if (ft == t) return true;
    }
    return false;
}

bool canScalarCast(BaseType from, BaseType to) {
    if (from == to) return true; // id
    switch (from) {
        case BaseType::BOOL:
            // '\0'/0/0.0 for false; 0x01/1/1.0 for true
            return to == BaseType::CHARACTER || to == BaseType::INTEGER || to == BaseType::REAL;
        case BaseType::CHARACTER:
            // false if '\0', true otherwise; ascii numeric for int/real
            return to == BaseType::BOOL || to == BaseType::INTEGER || to == BaseType::REAL;
        case BaseType::INTEGER:
            // false if 0, true otherwise; unsigned mod 256 for char; real version of integer
            return to == BaseType::BOOL || to == BaseType::CHARACTER || to == BaseType::REAL;
        case BaseType::REAL:
            // truncate for integer; real->bool/char are N/A
            return to == BaseType::INTEGER;
        default:
            return false;
    }
}

static bool canCastTypeImpl(const CompleteType& from, const CompleteType& to) {
    if (from.baseType == to.baseType) {
        // Shapes match + ensure all subtypes are mutually castable
        if (from.subTypes.size() != to.subTypes.size()) return false;
        for (size_t i = 0; i < from.subTypes.size(); ++i) {
            if (!canCastTypeImpl(from.subTypes[i], to.subTypes[i])) return false;
        }
        return true;
    }

    // Scalar to scalar per spec
    if (isScalarType(from.baseType) && isScalarType(to.baseType)) {
        return canScalarCast(from.baseType, to.baseType);
    }

    // Tuple to tuple: equal arity and pairwise scalar-castable
    if (from.baseType == BaseType::TUPLE && to.baseType == BaseType::TUPLE) {
        if (from.subTypes.size() != to.subTypes.size()) return false;
        for (size_t i = 0; i < from.subTypes.size(); ++i) {
            const auto& f = from.subTypes[i];
            const auto& t = to.subTypes[i];
            if (!(isScalarType(f.baseType) && isScalarType(t.baseType) && canScalarCast(f.baseType, t.baseType))) {
                return false;
            }
        }
        return true;
    }

    // TODO pt2: Support scalar -> array promotions and array<->array conversions (including
    // multi-dimensional) once CompleteType carries dimension/size metadata.
    // Until then, reject casts between composite non-tuple types unless shapes
    // are strictly identical (handled by the baseType==case above).
    return false;
}

bool canCastType(const CompleteType& from, const CompleteType& to) {
    return canCastTypeImpl(from, to);
}
