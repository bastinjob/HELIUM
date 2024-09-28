#include "lexer.h"
#include <iostream>

int main() {

    std::string sourceCode = "int x=10; \n string name = \"Helium\" ";

    Lexer lexer(sourceCode);

    Token token = lexer.getNextToken();
    while (token.getType() != TokenType::EndofFile){
        std::cout<<token.toString() << std::endl;
        token = lexer.getNextToken();
    }

    return 0;
}