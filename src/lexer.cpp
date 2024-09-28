#include "lexer.h"
#include <cctype>

Lexer::Lexer(const std::string &source)
    : source(source), start(0), current(0), line(1), column(0) {}


char Lexer::peek() const {
    if(isAtEnd()) return '\0';
    return source[current];
}

char Lexer::advance(){
    current++;
    column++;
    return source[current-1];
}

bool Lexer::isAtEnd() const {
    return current >= source.size();
}

Token Lexer::getNextToken(){
    while(!isAtEnd()) {
        start = current;
        char c = advance();

        //handling white spaces and new lines
        if(isspace(c)){
            if(c=='\n') {
                column=0;
            }
            continue;
        }

        //handle numbers
        if(isdigit(c)){
            return number();
        }
        //identifiers and keywords
        if(isalpha(c)){
            return identifier();
        }

        //operators
        if(ispunct(c)) {
            return operatorToken();
        }
        //finally invalid tokens
        return Token(TokenType::Invalid, std::string(1,c), line, column);

    }

    return Token(TokenType::EndofFile, "", line, column);

}

//handle alphanumerics
Token Lexer::identifier(){
    while(isalnum(peek())) advance(); 
    std::string value = source.substr(start, current-start);
    return Token(TokenType::Identifier, value, line, column);
}
//handle digits
Token Lexer::number() {
    while (isdigit(peek())) advance();
    std::string value = source.substr(start, current-start);
    return Token(TokenType::Number, value, line, column);
}
//handle operators (single character)
Token Lexer::operatorToken() {
    std::string value(1, source[current-1]);
    return Token(TokenType::Operator, value, line, column);
}