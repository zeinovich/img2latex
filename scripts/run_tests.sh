#!/bin/bash

GREEN='\033[1;32m'
NC='\033[0m'

echo -e "${GREEN}Running tests${NC}"
echo "----"
coverage run -m --data-file=../reports/.coverage pytest .
echo -e "${GREEN}Finished running tests${NC}"
echo "----"
echo -e "${GREEN}Coverage report${NC}"
echo "----"
coverage report --data-file=../reports/.coverage --show-missing