#ifndef TOKEN_H
#define TOKEN_H

#include <string>


//Enum for all possible token types

enum class TokenType {

    Identifier,
    Keyword,
    Number,
    Operator,
    String,
    EndofFile,
    Invalid

};

class Token {

public:
    Token(TokenType type, const std::string &value, int line, int column)
        : type(type), value(value), line(line), column(column) {}

    //Getters

    TokenType getType() const {return type;}
    const std::string &getValue() const {return value;}
    int getLine() const {return line;}
    int getColumn() const {return column;}

    std::string toString() const;

private:
    TokenType type;
    std::string value;
    int line, column;

};

#endif