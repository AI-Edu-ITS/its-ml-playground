# Lesson 1 Python Basic

We divide this section into this format:

- [Lesson 1 Python Basic](#lesson-1-python-basic)
  - [Your First Python Code](#your-first-python-code)
  - [Variables](#variables)
    - [Basic Variables](#basic-variables)
    - [How to Write Variables](#how-to-write-variables)
  - [Data Types](#data-types)
    - [Writing Data Types](#writing-data-types)
    - [Data Types Casting](#data-types-casting)

## Your First Python Code

Write a Hello, World! sentence is pretty easy in Python. You can type this syntax in Python interpreter or inside .py file to produce Hello, World! result

```python
print("Hello, World!") # Output: Hello, World!
```

Then, Python also provided two approach to define comment in its python file, namely single line comment and multi line comment. You can use `#` to mark single line as comment like following example:

```python
# print("Hello, World!)
```

From above example, Python will not produce anything since `print` statement is marked as comment. Comment also can be placed at the end of the line too.

To write multiline comment, You can use triple quotes in your code like following example:

```python
"""
This is only a comment
multi line command will not printed in console
just give it a try
"""
print("Hello, World!")
```

## Variables

### Basic Variables

To declare variable in Python, you just simply write this line:

```python
a = "Hello, World!"
print(a) # Output: Hello, World!
```

Different with other programming language, Python did not require any keyword to write variable. In addition, you did not need to write any data type before write your variable.

```python
str a = "hello, world!" # Will produce invalid syntax error
a = "hello, world!" # Output: hello, world!
```

### How to Write Variables

Python variable can be written in a single character (like `x` or `y`) or in a descriptive name (like `sum`, `name`, `phone_number`). There are some rules need to follow when writing a variable:

- Must start with a letter or underscore character
- Cannot start with number
- Only contain alpha-numeric character and underscores (`A-Z`, `a-z`, `0-9`, `_`)
- Python variable is case-sensitive, different case will point different variable (variable `age` is not same `Age`)
- Cannot be the name of Python keyword

## Data Types

Python supports multiple variables which commonly found in other programming language. You can refer to this table to use data types which support your need in writing Python code

| **Name** | **Types** | **Purpose** |
|:---------:|:--------:|:------------|
| `int` | Numeric | Hold non-decimal number of non-limited length |
| `long` | Numeric | Same as `int` but hold a longer length (deprecated in Python 3) |
| `float` | Numeric | Hold decimal number with accurate up to 15 decimal places |
| `complex` | Numeric | Hold complex numbers |
| `str` | String | Hold single or sequence of characters. Also support holding unicode characters |
| `list` | Sequence | Dynamic array version in Python, can hold multi data type as array |
| `tuple` | Sequence | Immutable array version in Python |
| `range` | Sequence | Hold sequence of numbers based on given value of start, stop, and step |
| `dict` | Mapping | Dictionary data type, hold key and value data types |
| `boolean` | Boolean | Hold `True` or `False` value |

### Writing Data Types

Here are an example of how to write each data type explained before

```python
a = 5 # int data type
b = 6 # long data type (deprecated in Python 3)
c = 4.3 # float data type
d = 100+3j # complex data type
e = "hello, world!" # str data type
f = [1, 0, 2] # list data type
g = (9, 8, 7) # tuple data type
h = range(5) # range data type, will produce [0, 1, 2, 3, 4]
i = {'a': 9, 'b': 10} # dict data type
j = False # boolean data type
```

You can get data type of any variable by using `type()` keyword like below

```python
a = 5
b = 3.4
print(type(a)) # Output: <class 'int'>
print(type(b)) # Output: <class 'float'>
```

### Data Types Casting

You can change a variable data type by using data type casting like following example

```python
x = str(2) # x value will be "2" instead of 2
y = float(3) # y value will be 3.0 instead of 3
```

However, some casting will not work in some cases like following example

```python
y = int("r5") # Will produce TypeError exception
z = float("4r") # Will produce ValueError exception
```
