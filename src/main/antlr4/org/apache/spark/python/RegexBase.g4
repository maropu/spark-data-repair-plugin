/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

grammar RegexBase;

regularExpression
    : CARET? expression DOLLAR?
    ;

expression
    : CONSTANT
	| ANY
	| expression STAR
	| expression PLUS
	| expression MAYBE
	| PATTERN
	| RANGE
	| expression expression
	| expression ALTERNATION expression
	;

PATTERN
	: '[' (CHARACTER | (CHARACTER '-' CHARACTER))+ ']'
	;

SYMBOL
    : CHARACTER
    | PATTERN
    ;

RANGE
    : SYMBOL '{' NUMBER '}'
	| SYMBOL '{' ',' NUMBER '}'
	| SYMBOL '{' NUMBER ',' '}'
	| SYMBOL '{' NUMBER ',' NUMBER '}'
	;

CONSTANT
    : (LETTER | DIGIT | ' ' | '_' | '-' | '%')+
    ;

STAR: '*';
PLUS: '+';
MAYBE: '?';
ALTERNATION: '|';
ANY: '.';
CARET: '^';
DOLLAR: '$';

CHARACTER
    : LETTER
    | DIGIT
    ;

NUMBER
    : DIGIT+
    ;

fragment DIGIT
    : [0-9]
    ;

fragment LETTER
    : [a-zA-Z]
    ;

WS
    : [\t\r\n]+ -> skip;