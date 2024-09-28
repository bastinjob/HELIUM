#ifndef LEXER_H
#define LEXER_H

#include "token.h"
#include <string.h>
#include <vector>

class Lexer{

public:
    //Contructor take source cod as input
    Lexer(const std::string &source);

    //Get next token
    Token getNextToken();

private:
    //define helper functions for different types of charcaters

    char peek() const;  //peek at the current character
    char advance();     //advance to the next character
    bool isAtEnd() const; //check if end of source file is reached

    Token identifier(); //Lex an identifier or keyword
    Token number();
    Token string();
    Token operatorToken();


    std::string source;
    int start;
    int current;
    int line;
    int column;

};

#endif