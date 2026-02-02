


the first thing we are convering is type checking type val;idation and type hintingin

the order of them is basically something like this

TYPE HINTING 
type hinting is like taking some extra notes when we might see some use in making the return type more obvious 
ie u want to tspecify u want the funcitoin to return a string not an int


def function_name(id) -> str:
    #body of the function 

what about parameters? 
well actually we can do the same thing

def function_name(id:str) -> str:
    #body of the function



TYPE CHECKING 
if type hinting was a footnote of whats expected then type checking is the costco attendabnt checking ur receipt
 if you did somehting like 

isInstance() 
# Basic type checks
isinstance(42, int)          # True
isinstance("hello", str)     # True
isinstance([1, 2, 3], list)  # True

# Check against multiple types
isinstance(42, (int, float))  # True

# Check for custom classes
class Person:
    pass

person = Person()
isinstance(person, Person)  # True

VALUE VALIDATION is essentially a way to check if a number is sensible 
somehting simple can be like for a regular standard plan user in the bank you could check if the etransfer is less than 10000 or not otherwise flag it as it is suspicious




the arrow for type hinting 

mypy and pydantic for data validation




MAIN LIBRARIES FOR THE ABOVE TOPICS:
## Core Built-in Libraries
- **`typing`** - Essential for type hints (annotations like `List[str]`, `Optional[int]`, etc.)

## Data Validation Libraries
- **`pydantic`** - Most popular for data validation with type hints. Automatically validates data based on type annotations and can generate schemas.

## Type Checking & Static Analysis
- **`mypy`** - Static type checker that works with your type hints to catch errors before runtime

## Testing & Data Checking
- **`pytest`** - Testing framework where you can write validation tests
- **`hypothesis`** - Property-based testing for generating test data and checking invariants

## Data Science Specific (for your fraud detection project)
- **`pandas`** - Data manipulation with built-in validation methods
- **`numpy`** - Numerical computing with type checking
- **`great_expectations`** - More advanced data validation for ML pipelines



SUMMARY IN MY OWN WORDS:
so basically pydantic is like using the info from the previously non functional type hints to use them to check if theya are coherent or something 
Yes, that's a good way to think about it. Pydantic takes Python's type hints (which are just annotations that don't enforce anything at runtime) and makes them functional for data validation.

from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    name: str
    age: Optional[int] = None
    email: str

# This validates and coerces data at runtime
user = User(name="John", age="25", email="john@example.com")
# age gets converted from str "25" to int 25
# If you pass invalid data, it raises ValidationError

# This would fail:
# User(name=123, email="invalid-email")  # TypeError + validation error




Mypy (Static Type Checker)
Primary focus: Checks type hints and type consistency
What it does: Verifies that your code adheres to type annotations without running it
Example: Catches int + str operations if types don't match

ok so basiaclly mypy isn't exactly a linter but it is a static analysis tool which wer can use for things like the operations between variables of different types like 5 - '1' which looks kind of ok at first glance but u know that 1 is interpreted as a string so ur lowey fkd



so....
next up on the learning list is data cleaning
where does it tie into everything 
well we've already been talking about potential methods 
of data cleaning so in a way its the overarching umbrella term that refers to things like data validation typechecking etc. 
other things that fall under data cleaning are data integrity, quality assurance and data prep 

WRONG IDEA
FEATURE ENGINEERING IS NOT SOMETHJING THAT FALLS UNDER DATA CLEANING, IT IS A STEP THAT COMES AFTER 



what about something like feature engineering 
yes its mainly broken down into 3 types like 
derived and composite fratures 
# Your example: sum column
df['sum'] = df['x'] + df['y']

# More complex: ratios, differences, products
df['ratio'] = df['x'] / df['y']
df['interaction'] = df['x'] * df['y']


ENCODING:
# Your RGB example
color_mapping = {'red': [255, 0, 0], 'green': [0, 255, 0], 'blue': [0, 0, 255]}

# Common techniques:
# One-hot encoding: 'red' → [1, 0, 0]
# Label encoding: 'red' → 0, 'green' → 1
# Target encoding: Use target variable statistics




FEATURE TRANFORMATION:
# Log transformation
df['log_amount'] = np.log1p(df['amount'])  # log(1+x) to handle zeros

# Other transformations:
df['sqrt_amount'] = np.sqrt(df['amount'])
df['boxcox_amount'], _ = stats.boxcox(df['amount'] + 1)