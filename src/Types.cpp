#include "Types.h"
#include "CompileTimeExceptions.h"
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
        case BaseType::EMPTY:      return "empty";
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
        case BaseType::EMPTY: return BaseType::EMPTY;
    }
    return BaseType::UNKNOWN;
}

CompleteType promote(const CompleteType& from, const CompleteType& to) {
    validateSubtypes(from);
    validateSubtypes(to);

    switch(from.baseType) {
        case BaseType::UNKNOWN:
            return CompleteType(BaseType::UNKNOWN);
        case BaseType::BOOL:
            switch(to.baseType) {
                case BaseType::BOOL:       return BaseType::BOOL;

                // Implicit scalar -> array-like promotions
                case BaseType::ARRAY: 
                case BaseType::VECTOR:
                case BaseType::MATRIX:
                {

                    // Need to be able to promote `from` type to array-like subtype
                    CompleteType subType = promote(from, to.subTypes[0]);

                    // Return UNKNOWN if subtype promotion failed
                    if (subType != to.subTypes[0]) return CompleteType(BaseType::UNKNOWN);

                    return to;
                };
                default: break;
            }
            break;
        case BaseType::CHARACTER:
            switch (to.baseType) {
                case BaseType::CHARACTER:  return BaseType::CHARACTER; 

                // Implicit scalar -> array-like promotions
                case BaseType::ARRAY: 
                case BaseType::VECTOR:
                case BaseType::MATRIX:
                {

                    // Need to be able to promote `from` type to array-like subtype
                    CompleteType subType = promote(from, to.subTypes[0]);

                    // Return UNKNOWN if subtype promotion failed
                    if (subType != to.subTypes[0]) return CompleteType(BaseType::UNKNOWN);

                    return to;
                };
                default: break;
            }
            break;
        case BaseType::INTEGER:
            switch (to.baseType) {
                case BaseType::INTEGER:    return BaseType::INTEGER; 
                case BaseType::REAL:       return BaseType::REAL;

                // Implicit scalar -> array-like promotions
                case BaseType::ARRAY: 
                case BaseType::VECTOR:
                case BaseType::MATRIX:
                {

                    // Need to be able to promote `from` type to array-like subtype
                    CompleteType subType = promote(from, to.subTypes[0]);

                    // Return UNKNOWN if subtype promotion failed
                    if (subType != to.subTypes[0]) return CompleteType(BaseType::UNKNOWN);

                    return to;
                };

                default: break;
            }
            break;
        case BaseType::REAL:
            switch (to.baseType) {
                case BaseType::REAL:       return BaseType::REAL;

                // Implicit scalar -> array-like promotions
                case BaseType::ARRAY: 
                case BaseType::VECTOR:
                case BaseType::MATRIX:
                {

                    // Need to be able to promote `from` type to array-like subtype
                    CompleteType subType = promote(from, to.subTypes[0]);

                    // Return UNKNOWN if subtype promotion failed
                    if (subType != to.subTypes[0]) return CompleteType(BaseType::UNKNOWN);

                    return to;
                };

                default: break;
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
                default: break;
            }
            break;
        case BaseType::STRING:
            switch (to.baseType) {
                case BaseType::STRING:     return BaseType::STRING;
                case BaseType::ARRAY: {
                    // STRING promotes to character[*] given length match
                    if (to.subTypes.size() != 1 || to.subTypes[0].baseType != BaseType::CHARACTER) {
                        return CompleteType(BaseType::UNKNOWN);
                    }
                    if (to.dims[0] != from.dims[0]) return CompleteType(BaseType::UNKNOWN);

                    return to; 
                }
                case BaseType::VECTOR: {
                    // STRING promotes to vector<char>
                    if (to.subTypes.size() != 1 || to.subTypes[0].baseType != BaseType::CHARACTER) {
                        return CompleteType(BaseType::UNKNOWN);
                    }
                    return to; 
                }

                default: break;
            }
            break;
        case BaseType::STRUCT:
            switch (to.baseType) {
                case BaseType::STRUCT: {
                    if (from.subTypes.size() != to.subTypes.size()) return CompleteType(BaseType::UNKNOWN);

                    CompleteType result(BaseType::STRUCT);
                    result.subTypes.reserve(from.subTypes.size());
                    for (size_t i = 0; i < from.subTypes.size(); ++i) {

                        // Structs do NOT support implicit field promotions
                        if (from.subTypes[i].baseType != to.subTypes[i].baseType) return CompleteType(BaseType::UNKNOWN);

                        result.subTypes.push_back(from.subTypes[i]);
                    }
                    return result;
                }
                default: break;
            }
            break;
        case BaseType::ARRAY:
            if (from.subTypes.size() != 1) {
                throw std::runtime_error(
                    "promote(): from array with subtype len " +
                    std::to_string(from.subTypes.size()));
            }

            switch (to.baseType) {
                case BaseType::ARRAY: 
                case BaseType::VECTOR:
                {
                    // ensure dimensions match (allow runtime len -1)
                    if (from.dims[0] != -1 && to.dims[0] != -1 && from.dims[0] != to.dims[0]) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    CompleteType subtypeResult = promote(from.subTypes[0], to.subTypes[0]);

                    if (subtypeResult != to.subTypes[0]) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    return to;
                }

                // Support promotions like array<integer> -> array<real> when mixing an array with a scalar
                case BaseType::INTEGER:
                case BaseType::REAL:
                {
                    if (from.dims.empty()) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    CompleteType elemType = from.subTypes[0];
                    CompleteType scalarType = to; // scalar target

                    // Try both promotion directions on the element types
                    CompleteType promotedElem = promote(elemType, scalarType);
                    if (promotedElem.baseType == BaseType::UNKNOWN) {
                        promotedElem = promote(scalarType, elemType);
                    }
                    if (promotedElem.baseType == BaseType::UNKNOWN) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    CompleteType result(BaseType::ARRAY);
                    result.subTypes.push_back(promotedElem);
                    result.dims = from.dims;
                    return result;
                }
                default: break;
            }
            break;
        case BaseType::VECTOR:
            if (from.subTypes.size() != 1) {
                throw std::runtime_error(
                    "promote(): from vector with subtype len " +
                    std::to_string(from.subTypes.size()));
            }

            switch (to.baseType) {
                case BaseType::ARRAY: 
                case BaseType::VECTOR:
                {
                    // ensure dimensions match (allow runtime len -1)
                    if (from.dims[0] != -1 && to.dims[0] != -1 && from.dims[0] != to.dims[0]) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    CompleteType subtypeResult = promote(from.subTypes[0], to.subTypes[0]);

                    if (subtypeResult != to.subTypes[0]) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    return to;
                }

                // Support promotions like vector<integer> -> vector<real>
                // when mixing a vector with a scalar. Compute the promoted
                // element type and keep the original length.
                case BaseType::INTEGER:
                case BaseType::REAL:
                {
                    if (from.dims.empty()) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    CompleteType elemType = from.subTypes[0];
                    CompleteType scalarType = to;

                    CompleteType promotedElem = promote(elemType, scalarType);
                    if (promotedElem.baseType == BaseType::UNKNOWN) {
                        promotedElem = promote(scalarType, elemType);
                    }
                    if (promotedElem.baseType == BaseType::UNKNOWN) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    CompleteType result(BaseType::VECTOR);
                    result.subTypes.push_back(promotedElem);
                    result.dims = from.dims;
                    return result;
                }
                default: break;
            }
            break;

        case BaseType::MATRIX:
            switch (to.baseType) {
                case BaseType::MATRIX: {
                    if (to.subTypes.size() != 1) {
                        throw std::runtime_error(
                            "promote(): to matrix with subtype len " +
                            std::to_string(to.subTypes.size()));
                    }

                    // ensure dimensions match
                    if (from.dims.size() != 2 || to.dims.size() != 2) {
                        return CompleteType(BaseType::UNKNOWN);
                    }
                    if (from.dims[0] != to.dims[0] || from.dims[1] != to.dims[1]) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    CompleteType result(BaseType::MATRIX);
                    CompleteType subtypeResult = promote(from.subTypes[0], to.subTypes[0]);
                    if (subtypeResult.baseType == BaseType::UNKNOWN) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    result.subTypes.push_back(subtypeResult);
                    result.dims = to.dims;

                    return result;
                }

                // Support promotions like matrix<integer> -> matrix<real>
                // when mixing a matrix with a scalar. Compute the promoted
                // element type and keep the original matrix dimensions.
                case BaseType::INTEGER:
                case BaseType::REAL:
                {
                    if (from.subTypes.size() != 1) {
                        throw std::runtime_error(
                            "promote(): from matrix with subtype len " +
                            std::to_string(from.subTypes.size()));
                    }
                    if (from.dims.size() != 2) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    CompleteType elemType = from.subTypes[0];
                    CompleteType scalarType = to;

                    CompleteType promotedElem = promote(elemType, scalarType);
                    if (promotedElem.baseType == BaseType::UNKNOWN) {
                        promotedElem = promote(scalarType, elemType);
                    }
                    if (promotedElem.baseType == BaseType::UNKNOWN) {
                        return CompleteType(BaseType::UNKNOWN);
                    }

                    CompleteType result(BaseType::MATRIX);
                    result.subTypes.push_back(promotedElem);
                    result.dims = from.dims;
                    return result;
                }

                default: break;
            }
            break;
        case BaseType::UNRESOLVED:
            throw std::runtime_error("Types::Promote: UNRESOLVED Type found");
            break;
        case BaseType::EMPTY: return BaseType::EMPTY;
    }
    return CompleteType(BaseType::UNKNOWN);
}

void validateSubtypes(CompleteType completeType) {
    // Flat type checking
    if (std::find(std::begin(flatTypes), std::end(flatTypes), completeType.baseType) != std::end(flatTypes)) {

        if (!completeType.subTypes.empty()) {
            throw std::runtime_error("Semantic Validation: Non-composite type of '" + toString(completeType.baseType) + "' cannot have subtypes.");
        }
        if (!completeType.dims.empty()) {
            throw std::runtime_error("Semantic Validation: Non-composite type of '" + toString(completeType.baseType) + "' cannot have dimensions.");
        }

    } else { // composite type validation
        if (completeType.baseType == BaseType::TUPLE && completeType.subTypes.size() < 2) {
            throw std::runtime_error("Semantic Validation: Tuple must have at least 2 subtypes.");
        }

        if (
            (completeType.baseType == BaseType::ARRAY ||
             completeType.baseType == BaseType::VECTOR ||
             completeType.baseType == BaseType::MATRIX) &&
            completeType.subTypes.size() != 1
        ) {
            throw std::runtime_error("Semantic Validation:" + toString(completeType.baseType) + " cannot have " + std::to_string(completeType.subTypes.size()) + " types.");
        }

        // Dimension constraints:
        //  - VECTOR: 1 dimension
        //  - ARRAY: 1  dimension
        //  - MATRIX: 2 dimensions
        // else: should not carry dimensions
        if (completeType.baseType == BaseType::VECTOR && completeType.dims.size() != 1) {
            throw std::runtime_error("Semantic Validation: Type" + toString(completeType) + " cannot have " + std::to_string(completeType.dims.size()) + " dimensions.");
        } else if (completeType.baseType == BaseType::ARRAY &&
                   (completeType.dims.empty() || completeType.dims.size() > 2)) {
            throw std::runtime_error("Semantic Validation: Type" + toString(completeType) + " cannot have " + std::to_string(completeType.dims.size()) + " dimensions.");
        } else if (completeType.baseType == BaseType::MATRIX && completeType.dims.size() != 2) {
            throw std::runtime_error("Semantic Validation: Type" + toString(completeType) + " cannot have " + std::to_string(completeType.dims.size()) + " dimensions.");
        } else if (isScalarType(completeType.baseType) || completeType.baseType == BaseType::TUPLE || completeType.baseType == BaseType::STRUCT){
            if (!completeType.dims.empty()) {
                throw std::runtime_error("Semantic Validation: Type" + toString(completeType) + " cannot have " + std::to_string(completeType.dims.size()) + " dimensions.");
            }
        }
        
        // Recursively validate nested subtypes
        for (const auto& subtype : completeType.subTypes) {
            validateSubtypes(subtype);
        }
    }
}

void validateContainmentHierarchy(const CompleteType& completeType, int line) {
    // UNRESOLVED/UNKNOWN/EMPTY are ignored here; they should be resolved or
    // rejected elsewhere in semantic analysis.
    if (completeType.baseType == BaseType::UNRESOLVED ||
        completeType.baseType == BaseType::UNKNOWN ||
        completeType.baseType == BaseType::EMPTY) {
        throw std::runtime_error("validateContainmentHierarchy: Invalid type " + toString(completeType) + " called as param");
    }

    // Scalars cant have sub-elements
    if (isScalarType(completeType.baseType)) {
        return;
    }

    // Enforce rules based on containment hierarchy
    //   tuple/struct
    //     -> vector/string
    //        -> 1D array / 2D array (matrix)
    //           -> bool/character/integer/real
    switch (completeType.baseType) {
        case BaseType::ARRAY:
        case BaseType::MATRIX: {
            // Arrays/matrices may only contain scalar element types.
            if (completeType.subTypes.size() != 1) {
                return; // structural issues handled by validateSubtypes
            }
            const CompleteType& elem = completeType.subTypes[0];
            if (!isScalarType(elem.baseType)) {
                throw TypeError(
                    line,
                    "Semantic Analysis: Invalid declared array element type");
            }

            validateContainmentHierarchy(elem, line);
            break;
        }
        case BaseType::VECTOR: {
            // Vectors may contain:
            //  - scalars
            //  - 1D arrays
            //  - 2D arrays/matrices
            if (completeType.subTypes.size() != 1) {
                return; // structural issues handled elsewhere
            }
            const CompleteType& elem = completeType.subTypes[0];
            const BaseType eb = elem.baseType;

            bool validSubtype =
                isScalarType(eb) || eb == BaseType::ARRAY || eb == BaseType::MATRIX;

            if (!validSubtype) {
                // Specifically rules out vector<struct>, vector<tuple>,
                // vector<vector>, vector<string>, etc.
                throw TypeError(
                    line,
                    "Semantic Analysis: Invalid declared array element type: " + toString(completeType));
                }
            validateContainmentHierarchy(elem, line);
            break;
        }
        case BaseType::TUPLE:
        case BaseType::STRUCT: {
            // Tuples/structs may not directly contain other tuples/structs.
            for (const auto& fieldType : completeType.subTypes) {
                if (fieldType.baseType == BaseType::STRUCT ||
                    fieldType.baseType == BaseType::TUPLE) {
                    throw TypeError(
                        line,
                        "Semantic Analysis: Invalid declared array element type: " + toString(completeType));
                }
                validateContainmentHierarchy(fieldType, line);
            }
            break;
        }
        case BaseType::STRING: {
            // Strings can be declared as 'string', no subtypes allowed
            if (!completeType.subTypes.empty()) {
                throw TypeError(
                    line,
                    "Semantic Analysis: Invalid declared string element type: " + toString(completeType));
            }
            break;
        }
        default: {
            for (const auto& sub : completeType.subTypes) {
                validateContainmentHierarchy(sub, line);
            }
            break;
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
    // Same top-level base type: require structurally compatible subtypes that are
    // themselves castable.
    if (from.baseType == to.baseType) {
        if (from.subTypes.size() != to.subTypes.size()) return false;
        for (size_t i = 0; i < from.subTypes.size(); ++i) {
            if (!canCastTypeImpl(from.subTypes[i], to.subTypes[i])) return false;
        }
        return true;
    }

    auto isAggregate = [](BaseType b) {
        return b == BaseType::ARRAY || b == BaseType::VECTOR || b == BaseType::MATRIX;
    };

    // Scalar to scalar 
    if (isScalarType(from.baseType) && isScalarType(to.baseType)) {
        return canScalarCast(from.baseType, to.baseType);
    }

    // Scalar -> aggregate broadcast 
    if (isScalarType(from.baseType) && isAggregate(to.baseType)) {
        if (to.subTypes.size() != 1) return false;
        const CompleteType &elem = to.subTypes[0];
        if (!isScalarType(elem.baseType)) return false;
        return canScalarCast(from.baseType, elem.baseType);
    }

    // Aggregate -> aggregate conversions for ARRAY/VECTOR/MATRIX
    if (isAggregate(from.baseType) && isAggregate(to.baseType)) {
        if (from.subTypes.size() != 1 || to.subTypes.size() != 1) return false;

        auto rankOf = [](const CompleteType &ct) -> std::size_t {
            return ct.dims.size();
        };

        std::size_t srcRank = rankOf(from);
        std::size_t dstRank = rankOf(to);

        // Backend only supports rank-1 and rank-2 aggregates here.
        if (srcRank == 0 || srcRank > 2 || dstRank == 0 || dstRank > 2) {
            return false;
        }

        // Element types must be recursively castable
        return canCastTypeImpl(from.subTypes[0], to.subTypes[0]);
    }
    return false;
}

bool canCastType(const CompleteType& from, const CompleteType& to) {
    return canCastTypeImpl(from, to);
}
