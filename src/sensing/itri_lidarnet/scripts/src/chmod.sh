#!/bin/bash

cd ../../

find . -type d -exec chmod 775 {} +

find . -type f -iname "*.md" -exec chmod -R  644 {} \;
find . -type f -iname "*.xml" -exec chmod -R  644 {} \;
find . -type f -iname "*.txt" -exec chmod -R  644 {} \;

find . -type f -iname "*.launch" -exec chmod -R  664 {} \;
find . -type f -iname "*.cpp" -exec chmod -R  664 {} \;
find . -type f -iname "*.h" -exec chmod -R  664 {} \;
find . -type f -iname "*.hpp" -exec chmod -R  664 {} \;
find . -type f -iname "*.msg" -exec chmod -R  664 {} \;
find . -type f -iname "*.yaml" -exec chmod -R  664 {} \;
find . -type f -iname "*.yml" -exec chmod -R  664 {} \;
find . -type f -iname "*.cc" -exec chmod -R  664 {} \;
find . -type f -iname "*.cu" -exec chmod -R  664 {} \;
find . -type f -iname "*.cuh" -exec chmod -R  664 {} \;
find . -type f -iname "*.cfg" -exec chmod -R  664 {} \;
find . -type f -iname "*.ui" -exec chmod -R  664 {} \;
