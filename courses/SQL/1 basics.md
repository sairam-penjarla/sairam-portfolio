## Introduction to SQL

SQL, which stands for Structured Query Language, is a programming language used to communicate with and manage databases. Think of a database as a big digital filing cabinet where data is stored in an organized manner. SQL allows you to interact with that data, making it easy to add, remove, update, and retrieve information in a structured way.

SQL is not a general-purpose programming language like Python or Java; instead, it is specifically designed to work with databases. With SQL, you can perform tasks like searching for specific information, sorting data, and even combining data from different places in the database. It's the most common language used in relational database management systems (RDBMS), such as MySQL, PostgreSQL, Microsoft SQL Server, and SQLite.

### What Can You Do with SQL?

- **Retrieve Data**: You can search for data based on specific conditions, sort it, or filter it.
- **Add Data**: You can insert new records or rows into a database.
- **Update Data**: You can modify existing data.
- **Delete Data**: You can remove data from a database.
- **Organize Data**: You can group data, calculate summaries, and perform mathematical operations.
- **Create and Manage Databases**: You can create databases, tables, and define how data should be stored.

SQL works by using a set of commands or statements. These commands are written in plain English-like syntax that is easy to understand. Some of the most common SQL commands include:

- **SELECT**: Used to retrieve data from a database.
- **INSERT**: Used to add new data.
- **UPDATE**: Used to modify existing data.
- **DELETE**: Used to remove data.

In SQL, data is usually organized into tables. Each table consists of rows and columns, similar to a spreadsheet, where each row is a record and each column holds a specific type of information (like name, age, or address). By using SQL queries, you can access, manipulate, and organize this data in many different ways.

SQL queries are composed of **clauses**, such as:

- **SELECT**: Specifies the columns you want to retrieve.
- **FROM**: Specifies the table from which you want to retrieve the data.
- **WHERE**: Defines conditions to filter the data.
- **ORDER BY**: Sorts the result set.

### Why SQL is Important

SQL is a fundamental skill for anyone working with databases. If you're an intern, a developer, or a data analyst, you’ll often need to extract, analyze, and manage data from databases. SQL makes these tasks simple and efficient.

SQL is used across various industries, from banking and healthcare to entertainment and social media, because it provides a standardized way to interact with databases. Knowing SQL allows you to work with data in almost any database system.

In the next lessons, you’ll learn how to use SQL commands to perform different tasks like retrieving data (using **SELECT**), filtering data (using **WHERE**), and even more complex tasks like joining data from multiple tables or creating views to make working with data easier.

SQL serves as the backbone for managing and querying databases, making it an essential skill for anyone working with data.

## **Data Types**

In SQL, **data types** define the kind of data that can be stored in a database column. Each column in a table must be assigned a data type, which determines what kind of values the column can hold. Understanding data types is important because it helps ensure that data is stored correctly and can be processed without errors.

### **Why Are Data Types Important?**

Data types serve several key purposes:

1. **Correct Data Storage**: Ensures that only valid types of data are stored in each column (e.g., no storing text in a column that should contain numbers).
2. **Efficient Data Management**: Different data types use different amounts of storage space, so choosing the right type helps keep the database efficient.
3. **Data Integrity**: Helps prevent mistakes, like entering a name in a column meant for numbers or entering a date in a text column.

SQL provides several built-in data types that can be categorized into different groups based on the type of data they store. Let’s look at some common data types:

### **Numeric Data Types**

1. **INT (Integer)**: Stores whole numbers (positive or negative) without any decimal places. For example, 1, -10, 1000 are all integers.
2. **DECIMAL (or NUMERIC)**: Used to store exact numeric values with a fixed number of decimal places. For example, 12.50 or -0.005. This is useful for storing financial data where precision is required.
3. **FLOAT (or REAL)**: Stores floating-point numbers (numbers with decimals), but it may not always be as precise as DECIMAL. It’s typically used for scientific calculations where small variations in precision are acceptable.
4. **BIGINT**: Used to store very large whole numbers. This is similar to INT but for much larger values.

### **Character/String Data Types**

1. **CHAR (Character)**: Used to store fixed-length strings (e.g., names, country codes). For example, if you define a column as `CHAR(10)`, it will always reserve space for 10 characters, even if the string is shorter.
2. **VARCHAR (Variable Character)**: Stores variable-length strings. This is more flexible than CHAR, as it only uses the amount of space needed for the actual data. For example, a VARCHAR(50) column can hold up to 50 characters but will only use as much space as needed for the string entered.
3. **TEXT**: Used to store large amounts of text, such as descriptions or long paragraphs. It can hold a significantly larger amount of data compared to VARCHAR.

### **Date and Time Data Types**

1. **DATE**: Stores dates in the format `YYYY-MM-DD`. For example, `2025-01-27` represents January 27, 2025. This type is used when you only care about the date (no time).
2. **TIME**: Stores time values in the format `HH:MM:SS`. For example, `14:30:00` represents 2:30 PM.
3. **DATETIME**: Combines both date and time into one data type. For example, `2025-01-27 14:30:00` stores both the date and the time.
4. **TIMESTAMP**: Similar to DATETIME, but it also includes the timezone information. This is often used to record the exact time an event happens (e.g., when a row is added to a table).

### **Boolean Data Type**

1. **BOOLEAN**: Stores only two possible values: true or false. This is useful for representing binary conditions, such as whether a user is active or not.

### **Binary Data Types**

1. **BLOB (Binary Large Object)**: Used to store binary data, such as images, videos, or files. This data type is for storing data that isn’t text but still needs to be stored in the database.

### **Other Data Types**

1. **ENUM**: Stores a list of predefined values. For example, a column could only have values like 'Small', 'Medium', or 'Large'. This is useful for fields that can only have a set of fixed values, like the status of an order (e.g., 'Pending', 'Shipped', 'Delivered').
2. **JSON**: Stores data in JSON format (a structured text format for representing data). This is useful for storing data that has a flexible structure, like objects or arrays, and is commonly used for web applications.

### **Choosing the Right Data Type**

Choosing the right data type is essential for several reasons:

- **Storage Efficiency**: Some data types, like VARCHAR, use only as much space as needed, while others, like CHAR, always use a fixed amount of space, even if it's not needed.
- **Performance**: Using appropriate data types can improve query performance because smaller and simpler data types are quicker to process.
- **Data Integrity**: Ensuring that data is stored in the correct type prevents errors and ensures that calculations and comparisons are accurate.

## **Statements**

In SQL, statements are used to modify and manage the data in your database. The three most common types of statements are `INSERT INTO`, `UPDATE`, and `DELETE`. Each of these allows you to insert new data, update existing data, or delete data, respectively. Let's start by creating a simple table and then look at how to use these statements to interact with that table.

### **Creating a Table**

First, let's create a table called **Employees** that we will use throughout this section.

```sql
-- Creating the Employees table to store employee information
CREATE TABLE Employees (
    EmployeeID INT,         -- Integer type for the unique Employee ID
    FirstName VARCHAR(50),  -- String type for the employee's first name
    LastName VARCHAR(50),   -- String type for the employee's last name
    Age INT,                -- Integer type for the employee's age
    Salary DECIMAL(10, 2)   -- Decimal type for storing the employee's salary with two decimal places
);

```

The above query creates the **Employees** table with columns for `EmployeeID`, `FirstName`, `LastName`, `Age`, and `Salary`.

### **INSERT INTO**

The `INSERT INTO` statement is used to add new records to a table. It’s commonly used when you need to input new data into your database.

```sql
-- Creating the Employees table (as shown above)
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting a new employee into the Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
VALUES (1, 'John', 'Doe', 35, 55000);

```

In this example, we are inserting a new employee with `EmployeeID` 1, the name "John Doe", age 35, and salary 55000 into the **Employees** table.

**Tips and Best Practices**:

- Always specify the column names in the `INSERT INTO` statement. This makes your query more readable and prevents errors if the table structure changes.
- When inserting data, make sure the values you provide match the data types of the columns in the table.
- You can insert multiple rows at once by separating the sets of values with commas:
    
    ```sql
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (2, 'Jane', 'Smith', 40, 60000),
           (3, 'Mike', 'Johnson', 28, 50000);
    
    ```
    

### **UPDATE**

The `UPDATE` statement is used to modify existing data in a table. You should always include a `WHERE` clause to specify which rows you want to update; otherwise, the update will affect all rows in the table.

```sql
-- Creating the Employees table (as shown above)
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Updating the salary of the employee with EmployeeID = 1
UPDATE Employees
SET Salary = 60000
WHERE EmployeeID = 1;

```

Here, we are updating the salary of the employee with `EmployeeID` 1 to 60000.

**Tips and Best Practices**:

- Always include a `WHERE` clause to specify the exact rows you want to update. Forgetting the `WHERE` clause will result in updating all the rows in the table, which can lead to data loss.
- If you need to update multiple columns at once, you can do so in a single `UPDATE` statement:
    
    ```sql
    UPDATE Employees
    SET Salary = 65000, Age = 36
    WHERE EmployeeID = 1;
    
    ```
    

### **DELETE**

The `DELETE` statement is used to remove records from a table. Like `UPDATE`, you should always include a `WHERE` clause to specify which rows to delete; otherwise, the command will remove all records in the table.

```sql
-- Creating the Employees table (as shown above)
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Deleting the employee with EmployeeID = 1
DELETE FROM Employees
WHERE EmployeeID = 1;

```

In this case, we are deleting the employee with `EmployeeID` 1 from the **Employees** table.

**Tips and Best Practices**:

- Always verify your `WHERE` clause before running a `DELETE` statement to avoid accidentally deleting all rows in the table.
- If you need to delete all rows in the table but keep the table structure intact, you can use the `TRUNCATE` statement instead:
This is faster than `DELETE` for large tables but should be used with caution.
    
    ```sql
    TRUNCATE TABLE Employees;
    
    ```
    

---

## Functions

### **SELECT**

The `SELECT` statement is used to retrieve data from one or more columns of a table. It’s one of the most commonly used SQL commands.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting sample data into the Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
VALUES (1, 'John', 'Doe', 35, 55000),
       (2, 'Jane', 'Smith', 28, 48000),
       (3, 'James', 'Bond', 45, 70000);

-- Using the SELECT statement to retrieve data
SELECT FirstName, LastName, Salary FROM Employees;

```

**Expected Output:**

```
| FirstName | LastName | Salary |
|-----------|----------|--------|
| John      | Doe      | 55000  |
| Jane      | Smith    | 48000  |
| James     | Bond     | 70000  |

```

---

### **WHERE**

The `WHERE` clause is used to filter records based on specific conditions.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting sample data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
VALUES (1, 'John', 'Doe', 35, 55000),
       (2, 'Jane', 'Smith', 28, 48000),
       (3, 'James', 'Bond', 45, 70000);

-- Using the WHERE clause to filter employees based on age
SELECT FirstName, LastName, Salary
FROM Employees
WHERE Age > 30;

```

**Expected Output:**

```
| FirstName | LastName | Salary |
|-----------|----------|--------|
| John      | Doe      | 55000  |
| James     | Bond     | 70000  |

```

---

### **INSERT INTO**

The `INSERT INTO` statement is used to insert new records into a table.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting new records into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
VALUES (1, 'John', 'Doe', 35, 55000),
       (2, 'Jane', 'Smith', 28, 48000);

```

---

### **UPDATE**

The `UPDATE` statement is used to modify existing records in a table.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting sample data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
VALUES (1, 'John', 'Doe', 35, 55000),
       (2, 'Jane', 'Smith', 28, 48000);

-- Updating the salary of employee with EmployeeID = 1
UPDATE Employees
SET Salary = 60000
WHERE EmployeeID = 1;

```

---

### **ORDER BY**

The `ORDER BY` clause is used to sort the result set based on one or more columns.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting sample data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
VALUES (1, 'John', 'Doe', 35, 55000),
       (2, 'Jane', 'Smith', 28, 48000),
       (3, 'James', 'Bond', 45, 70000);

-- Sorting employees by salary in descending order
SELECT FirstName, LastName, Salary
FROM Employees
ORDER BY Salary DESC;

```

**Expected Output:**

```
| FirstName | LastName | Salary |
|-----------|----------|--------|
| James     | Bond     | 70000  |
| John      | Doe      | 60000  |
| Jane      | Smith    | 48000  |

```

---

### **TRUNCATE**

The `TRUNCATE` statement is used to delete all rows from a table.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting sample data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
VALUES (1, 'John', 'Doe', 35, 55000),
       (2, 'Jane', 'Smith', 28, 48000);

-- Truncating the Employees table (removing all rows)
TRUNCATE TABLE Employees;

```

---

### **Aggregate Functions**

Aggregate functions perform calculations on a set of values and return a single value.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting sample data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
VALUES (1, 'John', 'Doe', 35, 55000),
       (2, 'Jane', 'Smith', 28, 48000),
       (3, 'James', 'Bond', 45, 70000);

-- Using aggregate functions to get total employees and average salary
SELECT COUNT(*) AS TotalEmployees, AVG(Salary) AS AverageSalary
FROM Employees;

```

**Expected Output:**

```
| TotalEmployees | AverageSalary |
|----------------|---------------|
| 3              | 57666.67      |

```

---

These examples cover the most essential SQL functions that a beginner should be familiar with. Each function is explained with a practical example, including how to create the table, insert data, and use the corresponding function to interact with the data.

## **Basic Operators**

### **Arithmetic Operators**

Arithmetic operators allow you to perform basic mathematical operations on numeric data types.

1. **Addition (+)**
    
    ```sql
    -- Creating a table for Employees with Salary column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Performing Addition operation on Salary
    SELECT FirstName, LastName, Salary, Salary + 5000 AS NewSalary
    FROM Employees;
    
    ```
    
    **Expected Output:**
    
    ```
    | FirstName | LastName | Salary | NewSalary |
    |-----------|----------|--------|-----------|
    | John      | Doe      | 50000  | 55000     |
    | Jane      | Smith    | 55000  | 60000     |
    | James     | Bond     | 60000  | 65000     |
    
    ```
    
2. **Subtraction (-)**
    
    ```sql
    -- Creating a table for Employees with Salary column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Performing Subtraction operation on Salary
    SELECT FirstName, LastName, Salary, Salary - 1000 AS NewSalary
    FROM Employees;
    
    ```
    
    **Expected Output:**
    
    ```
    | FirstName | LastName | Salary | NewSalary |
    |-----------|----------|--------|-----------|
    | John      | Doe      | 50000  | 49000     |
    | Jane      | Smith    | 55000  | 54000     |
    | James     | Bond     | 60000  | 59000     |
    
    ```
    
3. **Multiplication (*)**
    
    ```sql
    -- Creating a table for Employees with Salary column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Performing Multiplication operation on Salary
    SELECT FirstName, LastName, Salary, Salary * 1.1 AS NewSalary
    FROM Employees;
    
    ```
    
    **Expected Output:**
    
    ```
    | FirstName | LastName | Salary | NewSalary |
    |-----------|----------|--------|-----------|
    | John      | Doe      | 50000  | 55000     |
    | Jane      | Smith    | 55000  | 60500     |
    | James     | Bond     | 60000  | 66000     |
    
    ```
    
4. **Division (/)**
    
    ```sql
    -- Creating a table for Employees with Salary column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Performing Division operation on Salary
    SELECT FirstName, LastName, Salary, Salary / 2 AS HalfSalary
    FROM Employees;
    
    ```
    
    **Expected Output:**
    
    ```
    | FirstName | LastName | Salary | HalfSalary |
    |-----------|----------|--------|------------|
    | John      | Doe      | 50000  | 25000      |
    | Jane      | Smith    | 55000  | 27500      |
    | James     | Bond     | 60000  | 30000      |
    
    ```
    
5. **Modulus (%)**
    
    ```sql
    -- Creating a table for Employees with Salary column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Performing Modulus operation on Salary
    SELECT FirstName, LastName, Salary, Salary % 2 AS Remainder
    FROM Employees;
    
    ```
    
    **Expected Output:**
    
    ```
    | FirstName | LastName | Salary | Remainder |
    |-----------|----------|--------|-----------|
    | John      | Doe      | 50000  | 0         |
    | Jane      | Smith    | 55000  | 0         |
    | James     | Bond     | 60000  | 0         |
    
    ```
    

### **Comparison Operators**

Comparison operators are used to compare two values. These are most commonly used in the `WHERE` clause to filter data based on conditions.

1. **Equal to (=)**
    
    ```sql
    -- Creating a table for Employees with Age column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Comparing Age to a specific value
    SELECT * FROM Employees
    WHERE Age = 30;
    
    ```
    
    **Expected Output:**
    
    ```
    | EmployeeID | FirstName | LastName | Age | Salary |
    |------------|-----------|----------|-----|--------|
    | 1          | John      | Doe      | 30  | 50000  |
    
    ```
    
2. **Not equal to (<> or !=)**
    
    ```sql
    -- Creating a table for Employees with Age column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Comparing Age to ensure it's not equal to 30
    SELECT * FROM Employees
    WHERE Age <> 30;
    
    ```
    
    **Expected Output:**
    
    ```
    | EmployeeID | FirstName | LastName | Age | Salary |
    |------------|-----------|----------|-----|--------|
    | 2          | Jane      | Smith    | 28  | 55000  |
    | 3          | James     | Bond     | 35  | 60000  |
    
    ```
    
3. **Greater than (>)**
    
    ```sql
    -- Creating a table for Employees with Salary column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Comparing Salary to be greater than 50,000
    SELECT * FROM Employees
    WHERE Salary > 50000;
    
    ```
    
    **Expected Output:**
    
    ```
    | EmployeeID | FirstName | LastName | Age | Salary |
    |------------|-----------|----------|-----|--------|
    | 2          | Jane      | Smith    | 28  | 55000  |
    | 3          | James     | Bond     | 35  | 60000  |
    
    ```
    
4. **Less than (<)**
    
    ```sql
    -- Creating a table for Employees with Salary column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Comparing Salary to be less than 50,000
    SELECT * FROM Employees
    WHERE Salary < 50000;
    
    ```
    
    **Expected Output:**
    
    ```
    | EmployeeID | FirstName | LastName | Age | Salary |
    |------------|-----------|----------|-----|--------|
    | 1          | John      | Doe      | 30  | 50000  |
    
    ```
    
5. **Greater than or equal to (>=)**
    
    ```sql
    -- Creating a table for Employees with Age column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
    (2, 'Jane', 'Smith', 28, 55000),
    (3, 'James', 'Bond', 35, 60000);
    
    - - Comparing Age to ensure it's greater than or equal to 30
    SELECT * FROM Employees
    WHERE Age >= 30;
    
    ```
    
    Expected Output:
    
    ```
    | EmployeeID | FirstName | LastName | Age | Salary |
    | --- | --- | --- | --- | --- |
    | 1 | John | Doe | 30 | 50000 |
    | 3 | James | Bond | 35 | 60000 |
    
    ```
    
    6. Less than or equal to (<=)
    
    ```
    -- Creating a table for Employees with Age column
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT,
        Salary DECIMAL(10, 2)
    );
    
    -- Inserting sample data into Employees table
    INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
    VALUES (1, 'John', 'Doe', 30, 50000),
           (2, 'Jane', 'Smith', 28, 55000),
           (3, 'James', 'Bond', 35, 60000);
    
    -- Comparing Age to ensure it's less than or equal to 30
    SELECT * FROM Employees
    WHERE Age <= 30;
    
    ```
    
    **Expected Output:**
    
    ```
    | EmployeeID | FirstName | LastName | Age | Salary |
    |------------|-----------|----------|-----|--------|
    | 1          | John      | Doe      | 30  | 50000  |
    | 2          | Jane      | Smith    | 28  | 55000  |
    
    ```
    
    This approach shows how the `INSERT` statements populate data into the tables, followed by `SELECT` queries that display the results based on different operators. 
    
    ---