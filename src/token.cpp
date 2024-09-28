#include "token.h"
#include <sstream>


std::string Token::toString() const {

    std::stringstream ss;
    ss << "Token(" << static_cast<int>(type) << ", '" << value << "', Line: " <<line << ", Column: " <<column << ")";

    return ss.str();
}