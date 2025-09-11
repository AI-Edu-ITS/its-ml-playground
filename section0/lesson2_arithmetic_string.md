# Lesson 2 Operators & String Operation

We divide this section into this format:

- [Lesson 2 Operators \& String Operation](#lesson-2-operators--string-operation)
  - [About Operators](#about-operators)
    - [Arithmetic Operators](#arithmetic-operators)
    - [Bitwise Operator](#bitwise-operator)
    - [Logical Operators](#logical-operators)
    - [Asignment Operators](#asignment-operators)
    - [Comparison Operators](#comparison-operators)
    - [Identity Operators](#identity-operators)
    - [Membership Operators](#membership-operators)
  - [String Operation](#string-operation)
    - [String Slicing](#string-slicing)
    - [String Modification](#string-modification)
    - [String Formatting](#string-formatting)

## About Operators

Python has numerous operators to use when processing variables and values. There are 7 operators type in Python which divided as below.

### Arithmetic Operators

These operators are used to perform mathematic operation.

| **Operator** | **Name** |
| :----------- | :------- |
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `%` | Modulus |
| `**` | Exponentiation |
| `//` | Floor Division |

### Bitwise Operator

These operators are used to compare numbers in form of binary.

| **Operator** | **Name** | **Description** |
| :----------- | :------- | :-------------- |
| `&` | AND | Set each bit to 1 if both bits are 1 |
| `\|` | OR | Set each bit to 1 if one of two bits is 1 |
| `^` | XOR | Sets each bit to 1 if only one of two bits is 1 |
| `~` | NOT | Inverts all the bits |
| `<<` | Zero Fill Left Shift | Shift left by pushing zeros in from the right and let the leftmost bits fall off |
| `>>` | Signed Right Shift | Shift right by pushing copies of the leftmost bit in from the left, and let the rightmost bits fall off |

To give an understanding of these operators, lets see an example below.

```python
x = 119
y = 124
print(x & y) # Output: 116
print(x | y) # Output: 127
print(x ^ y) # Output: 11
print(~x) # Output: -120
```

Then, we turn `x` and `y` into binary and apply bitwise operator in each operation

| **x binary** | **y binary** | **x & y** | **x \| y** | **x ^ y** |
| :----------: | :----------: | :-------: | :--------: | :-------: |
| 1 | 1 | 1 | 1 | 0 |
| 1 | 1 | 1 | 1 | 0 |
| 1 | 1 | 1 | 1 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |

Where `1110100` is equal to `116`, `1111111` is equal to `127`, and `0001011` is equal to `11`.

To conduct bitwise NOT (~) operation, we use equation `~x = (-1 * (x+1))`. Suppose we use `119` as value we want to apply bitwise NOT operation, first we transform `119` into binary form which resulting into `1110111`. Then we conduct addition with `1` resulting to `1110000`. Then we multiply it with -1 which produce `-1110000`. If we convert to decimal, the final result is `-120`.

### Logical Operators

These operators are usually used to combine conditional statement declaration.

| **Operator** | **Name** |
| :----------- | :------- |
| `and` | Returns True if both statements are true |
| `or` | Returns True if one of the statements is true |
| `not` | Reverse the result, returns False if the result is true and vice versa |

### Asignment Operators

These operators are used to assign values to variables.

| **Operator** | **Name** | **Example** | **Similar To** |
| :----------- | :------- | :---------- | :------------- |
| `=` | Simple Assignment | `y = 1` | `y = 1` |
| `:=` | Walrus Operator | `y := 1` | `y = 1` |
| `+=` | Add & Assign | `y += 1` | `y = y + 1` |
| `-=` | Subtract & Assign | `y -= 1` | `y = y - 1` |
| `*=` | Multiply & Assign | `y *= 1` | `y = y * 1` |
| `/=` | Divide & Assign | `y /= 1` | `y = y / 1` |
| `%=` | Modulus & Assign | `y %= 1` | `y = y % 1` |
| `//=` | Floor & Assign | `y //= 1` | `y = y // 1` |
| `**=` | Exponent & Assign | `y **= 1` | y = y ** 1 |
| `&=` | Bitwise AND & Assign | `y &= 1` | `y = y & 1` |
| `\|=` | Bitwise OR & Assign | `y \|= 1` | `y = y \| 1` |
| `^=` | Bitwise XOR & Assign | `y ^= 1` | `y = y ^ 1` |
| `>>=` | Zero Fill Left Shift & Assign | `y >>= 1` | `y = y >> 1` |
| `<<=` | Signed Right Shift & Assign | `y <<= 1` | `y = y << 1` |

### Comparison Operators

These operators are used to compare two given values.

| **Operator** | **Name** |
| :----------- | :------- |
| `==` | Equal |
| `!=` | Not Equal |
| `>` | Greater Than |
| `<` | Less Than |
| `>=` | Greater Than or Equal To |
| `<=` | Less Than or Equal To |

### Identity Operators

These operators are used to compare two objects with the same memory location.

| **Operator** | **Name** |
| :----------- | :------- |
| `is` | Returns True if both variables are the same object |
| `is not` | Returns True if both variables are not the same object |

### Membership Operators

These operators are used to test if a sequence is presented in an object.

| **Operator** | **Name** |
| :----------- | :------- |
| `in` | Returns True if a sequence with the specified value is present in the object |
| `not in` | Returns True if a sequence with the specified value is not present in the object |

## String Operation

### String Slicing

You can return string combination using slicing like in array, you need to define the start index of string and the end ondex of string.

```python
my_str = "Hello, World!"
print(my_str[3:6]) # Output: 'lo,'
```

In above example, Python will produce string from exactly its given start index and exactly **before** the end index instead of printing its end index string too.

If you wanted to include some string ranging from n-index until end of string or from start of the string until exactly before n-index string, you can use following code.

```python
my_str = "Hello, World!"
print(my_str[4:]) # Output: 'o, World!'
print(my_str[:4]) # Output: 'Hell'
```

### String Modification

You can make a string into lowercase using `lower()` keyword and or into uppercase using `upper()` keyword.

```python
my_str = "Hello, World!"
print(my_str.lower()) # Output: 'hello, world!'
print(my_str.upper()) # Output: 'HELLO, WORLD!'
```

You can also split a string into array of string based on given delimiter using `split()` keyword.

```python
my_str = "Hello, World!"
print(my_str.split(",")) # Output: ['Hello', ' World!']
```

### String Formatting

To add variable value into a string, you can use `f` keyword right at the start of string like below.

```python
price = 100
my_str = f"This laptop price is {price} dollar"
print(my_str) # Output: "This laptop price is 100 dollar"
```

You can also add modifiers when formatting value in string.

```python
price = 100
my_str = f"This laptop price is {price:.3f} dollar"
print(my_str) # Output: "This laptop price is 100.000 dollar"
```

Or you can add mathematical operation when formatting value inside a string.

```python
price = 100
quantity = 5
my_str = f"This laptop price is {price * quantity} dollar"
print(my_str) # Output: "This laptop price is 500 dollar"
```
