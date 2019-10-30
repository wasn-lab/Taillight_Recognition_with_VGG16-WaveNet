# Coding style guideline

Coding style 是軟體開發時一個重要的工作，根據經驗，看程式碼的時間會遠比寫程式碼的時間來得長，一份乾淨、風格一致的程式碼可以讓人在無形中省去理解上的難度，也使之更容易維護和除錯。

針對既有的程式碼，若其coding style不符此文件之描述的話，先暫時不要去改動，等到手邊有足夠空檔能夠處理時，再做coding style的調整。

這個文件主要參考[ROS的coding style](http://wiki.ros.org/CppStyleGuide)，
由於空白、括號、縮排等風格可以由[透過程式自動排版](https://github.com/davetcoleman/roscpp_code_format)，
這裡只描述Naming的部分

## 命名的方式

命名的方式有以下幾種：

* CamelCased: 命名由多個單字構成，每個單字的首字母大寫，其餘小寫。
* camelCased: 像CamelCase的風格，但第一個字母個小寫。
* under_scored: 只使用小寫字母，每個單字之間用底線隔開。
* ALL_CAPITALS: 只使用大寫字母，每個單字之間用底線隔開。

針對不同的對象，使用不同的命名風格

* package 的命名： 使用 **under_scored** 之風格。
* Topics/Services 的命名：使用 **under_scored** 之風格。
* 檔名：所有檔案都是 **under_scored** 之風格，source code的副檔名為.cpp, header 檔的副檔名為.h, 若檔案內容主體為實作class的話，以其class作命名；如實作ActionServer class的檔案應命名為action_server.cpp和action_server.h。
* library 的命名：library也是檔案的一種，使用*under_scored** 之風格。
* class/type的命名：使用CamelCased的風格，如
```
class ExampleClass;
```
但若class的名字裡含有縮寫，其縮寫應全部為大寫，如
```
class HokuyoURGLaser;
```
* function/methods的命名：function及method的命名使用**camelCased**的風格，其參數使用**under_scored**風格，如
```
int exampleMethod(int example_arg);
```
function和method通常是執行一個動作，所以使用動詞來命名會比較好，如checkForErrors() 比 errorCheck() 來得好, dumpDataToFile() 比 dataFile() 來得好；相對的，class通常是名詞。

* 變數名稱：使用**under_scored**風格
* 常數：使用**ALL_CAPITALS**風格
* Member variables: 使用**under_scored**風格，並在結尾加一個底線，如
```
int example_int_;
```
* global variable：儘量避免使用global variable, 若非得要用，則使用**under_scored**風格並在最前面加 g_，如
```
// I tried everything else, but I really need this global variable
int g_shutdown;
```
* namespace：使用**under_scored**風格。

## header guard

所有header都要用#ifdef保護起來，以免重覆include, 如
```
#ifndef PACKAGE_PATH_FILE_H
#define PACKAGE_PATH_FILE_H
...
#endif
```

## Console output

待補充

## Macros

避免使用Macro，理由是debug symbol裡沒有macro的資訊，難以debug, 另macro容易被過度使用，不容易看出其運作方式。

## Preprocess directive

若程式中有conditional compilation, 使用 #if，不要使用 #ifdef，如
```
#if DEBUG
  temporary_debugger_break();
#endif
```

## output argument

function/method的output argument應使用pointer 傳入，而不是使用reference傳入，如
```
int exampleMethod(FooThing input, BarThing* output);
```

理由參閱[Reference Arguments](https://google.github.io/styleguide/cppguide.html#Reference_Arguments)

## namespace

鼓勵使用具有描述性的namespace

header裡不要使用using-directive, 因為這會影響到所有include該header的檔案。

儘量使用using-declarations而非using-directives，因為這麼做只會把會使用到的東西包含進來，例如
```
using std::list;  // I want to refer to std::list as list
using std::vector;  // I want to refer to std::vector as vector
```
不好的寫好：
```
using namespace std; // Bad, because it imports all names from std::
```

## 繼承

base class 定義介面，subclass實作之。

繼承可以為base class和subclass提供相同的程式碼，但不鼓勵使用這種繼承方式，在多數情況下，subclass可以用base class的instance達到相同的目的且不會造成混淆。

subclass要對virtual method做override時，一定要把它宣告成virtual，讀者才明白實際發生什麼事。

## 多重繼承

避免使用多重繼承，因為極容易讓人混淯。

## Exceptions

**不要**在destructor中throw exceptions.
**不要**在callback中throw exceptions, 除非你直接呼叫它。

若程式中另有使用error code做error handling, 就僅使用error code，讓風格一致。

## Enumerations

在namespace中使用enums, 如
```
namespace Choices
{
  enum Choice
  {
     Choice1,
     Choice2,
     Choice3
  };
}
typedef Choices::Choice Choice;
```
這麼做可以保護程式碼不被enum汙染。

## Globals

避免使用global variable及global function, 尤其是global variable, 它會讓multi-thread程式變得難以駕御。

大多數的變數或function都可以被宣告在class裡面，其餘的可以在namespace裡宣告。

## static class variables

避免使用static class variable, 它會讓multi-thread程式變得難以駕御。


## 呼叫 exit()

除了適當的程式結束點以外，不要使用exit()。

在library中不可使用exit()。

## assertions

不要直接使用assert(), 應使用ros/assert.h裡的
```
/** ROS_ASSERT asserts that the provided expression evaluates to
 * true.  If it is false, program execution will abort, with an informative
 * statement about which assertion failed, in what file.  Use ROS_ASSERT
 * instead of assert() itself.
 * Example usage:
 */
   ROS_ASSERT(x > y);

/** ROS_ASSERT_MSG(cond, "format string", ...) asserts that the provided
 * condition evaluates to true.
 * If it is false, program execution will abort, with an informative
 * statement about which assertion failed, in what file, and it will print out
 * a printf-style message you define.  Example usage:
 */
   ROS_ASSERT_MSG(x > 0, "Uh oh, x went negative.  Value = %d", x);

/** ROS_ASSERT_CMD(cond, function())
 * Runs a function if the condition is false. Usage example:
 */
   ROS_ASSERT_CMD(x > 0, handleError(...));

/** ROS_BREAK aborts program execution, with an informative
 * statement about which assertion failed, in what file. Use ROS_BREAK
 * instead of calling assert(0) or ROS_ASSERT(0). You can step over the assert
 * in a debugger.
 * Example usage:
 */
   ROS_BREADK();
```

assert裡只檢查條件，不要做運算，如
```
ROS_ASSERT(x++ > 0); // BAD
```

在Release build的情況下，會加進-DNDEBUG的參數給compiler, 上述之assert會因此而失效，不會對程式的效能造成影響。

## portability

讓程式保持portable
* 不要使用uint, 而應使用unsigned int
* 在std namespace裡呼叫isnan(), 即std::isnan()
