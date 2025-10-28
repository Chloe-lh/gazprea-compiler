#include "Types.h"
#include <sstream>

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
    }
    return "unknown";
}

std::string toString(const CompleteType& type) {
    std::ostringstream oss;
    oss << toString(type.baseType);

    if (!type.subTypes.empty()) {
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
