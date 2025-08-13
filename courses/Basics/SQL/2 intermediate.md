## **Constraints and Normalization**

In SQL, constraints are used to enforce rules and limits on the data in your database tables. Constraints help maintain data integrity, ensuring that the data is valid and accurate. These constraints can be applied to individual columns or to the entire table. Let’s first understand the different types of constraints.

### **Types of Constraints**

1. **NOT NULL**
    - The `NOT NULL` constraint ensures that a column cannot have a `NULL` value. It is used when you want to make sure that every row in the table has a valid entry for a given column.
    
    ```sql
    -- Creating a table with NOT NULL constraint
    CREATE TABLE Employees (
        EmployeeID INT NOT NULL,  -- EmployeeID cannot be NULL
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT
    );
    
    ```
    
2. **UNIQUE**
    - The `UNIQUE` constraint ensures that all values in a column are distinct, meaning no duplicates are allowed.
    
    ```sql
    -- Creating a table with UNIQUE constraint
    CREATE TABLE Employees (
        EmployeeID INT,
        FirstName VARCHAR(50),
        LastName VARCHAR(50) UNIQUE,  -- LastName must be unique
        Age INT
    );
    
    ```
    
3. **PRIMARY KEY**
    - The `PRIMARY KEY` constraint uniquely identifies each record in a table. It is a combination of the `NOT NULL` and `UNIQUE` constraints. Each table can have only one primary key, and it must consist of one or more columns.
    
    ```sql
    -- Creating a table with PRIMARY KEY constraint
    CREATE TABLE Employees (
        EmployeeID INT PRIMARY KEY,  -- EmployeeID is the unique identifier for each employee
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT
    );
    
    ```
    
4. **FOREIGN KEY**
    - The `FOREIGN KEY` constraint is used to link two tables together. It ensures that the value in a column matches a value in the referenced column of another table. This helps maintain relationships between tables.
    
    ```sql
    -- Creating two tables with FOREIGN KEY constraint
    CREATE TABLE Departments (
        DepartmentID INT PRIMARY KEY,
        DepartmentName VARCHAR(50)
    );
    
    CREATE TABLE Employees (
        EmployeeID INT PRIMARY KEY,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        DepartmentID INT,
        FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID)
    );
    
    ```
    
5. **CHECK**
    - The `CHECK` constraint ensures that all values in a column satisfy a specific condition. It is often used to enforce domain integrity.
    
    ```sql
    -- Creating a table with CHECK constraint
    CREATE TABLE Employees (
        EmployeeID INT PRIMARY KEY,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT CHECK (Age >= 18)  -- Ensures that age is greater than or equal to 18
    );
    
    ```
    
6. **DEFAULT**
    - The `DEFAULT` constraint provides a default value for a column when no value is specified during insertion.
    
    ```sql
    -- Creating a table with DEFAULT constraint
    CREATE TABLE Employees (
        EmployeeID INT PRIMARY KEY,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        Age INT DEFAULT 30  -- If no age is provided, 30 will be used as the default
    );
    
    ```
    

### **Normalization**

Normalization is the process of organizing data in a way that reduces redundancy and dependency. It involves dividing large tables into smaller ones and defining relationships between them to ensure data integrity.

1. **First Normal Form (1NF)**
    - A table is in **First Normal Form** if all columns contain atomic (indivisible) values, and each record is unique.
    
    ```sql
    -- Example of a table that is not in 1NF
    CREATE TABLE Orders (
        OrderID INT,
        CustomerName VARCHAR(50),
        Products VARCHAR(100)  -- This column contains multiple products, which violates 1NF
    );
    
    -- To convert it to 1NF, we would split the Products column into separate rows
    CREATE TABLE Orders (
        OrderID INT,
        CustomerName VARCHAR(50),
        ProductName VARCHAR(50)  -- Each product is now in its own row
    );
    
    ```
    
2. **Second Normal Form (2NF)**
    - A table is in **Second Normal Form** if it is in 1NF and all non-key columns are fully dependent on the primary key. This means there should be no partial dependency of any column on the primary key.
    
    ```sql
    -- Example of a table that is not in 2NF
    CREATE TABLE Orders (
        OrderID INT,
        CustomerName VARCHAR(50),
        ProductName VARCHAR(50),
        ProductPrice DECIMAL(10, 2),
        PRIMARY KEY (OrderID, ProductName)
    );
    
    -- To convert it to 2NF, we would remove partial dependency by creating separate tables
    CREATE TABLE Orders (
        OrderID INT,
        CustomerName VARCHAR(50),
        PRIMARY KEY (OrderID)
    );
    
    CREATE TABLE Products (
        ProductName VARCHAR(50),
        ProductPrice DECIMAL(10, 2),
        PRIMARY KEY (ProductName)
    );
    
    ```
    
3. **Third Normal Form (3NF)**
    - A table is in **Third Normal Form** if it is in 2NF and all columns are not transitively dependent on the primary key. This means there are no non-key columns that depend on other non-key columns.
    
    ```sql
    -- Example of a table that is not in 3NF
    CREATE TABLE Orders (
        OrderID INT,
        CustomerName VARCHAR(50),
        CustomerCity VARCHAR(50),
        ProductName VARCHAR(50),
        PRIMARY KEY (OrderID)
    );
    
    -- To convert it to 3NF, we would remove transitive dependencies by creating a separate table for customers
    CREATE TABLE Orders (
        OrderID INT,
        CustomerID INT,
        ProductName VARCHAR(50),
        PRIMARY KEY (OrderID)
    );
    
    CREATE TABLE Customers (
        CustomerID INT,
        CustomerName VARCHAR(50),
        CustomerCity VARCHAR(50),
        PRIMARY KEY (CustomerID)
    );
    
    ```
    

## Joins

### **Creating a Table**

Let's start by creating two tables: **Employees** and **Departments**. We’ll use these tables to demonstrate different types of joins.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT
);

-- Creating the Departments table
CREATE TABLE Departments (
    DepartmentID INT,
    DepartmentName VARCHAR(50)
);

```

---

### **INNER JOIN**

An `INNER JOIN` returns records that have matching values in both tables.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT
);

-- Creating the Departments table
CREATE TABLE Departments (
    DepartmentID INT,
    DepartmentName VARCHAR(50)
);

-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID)
VALUES (1, 'John', 'Doe', 1),
       (2, 'Jane', 'Smith', 2),
       (3, 'James', 'Bond', 3);

-- Inserting data into Departments table
INSERT INTO Departments (DepartmentID, DepartmentName)
VALUES (1, 'HR'),
       (2, 'IT'),
       (3, 'Finance');

-- Using INNER JOIN to get employees and their department names
SELECT Employees.FirstName, Employees.LastName, Departments.DepartmentName
FROM Employees
INNER JOIN Departments
ON Employees.DepartmentID = Departments.DepartmentID;

```

**Expected Output:**

```
| FirstName | LastName | DepartmentName |
|-----------|----------|----------------|
| John      | Doe      | HR             |
| Jane      | Smith    | IT             |
| James     | Bond     | Finance        |

```

---

### **LEFT JOIN**

A `LEFT JOIN` returns all records from the left table (Employees), along with matching records from the right table (Departments). If there’s no match, NULL values will be returned for the right table.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT
);

-- Creating the Departments table
CREATE TABLE Departments (
    DepartmentID INT,
    DepartmentName VARCHAR(50)
);

-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID)
VALUES (1, 'John', 'Doe', 1),
       (2, 'Jane', 'Smith', 2),
       (3, 'James', 'Bond', NULL);

-- Inserting data into Departments table
INSERT INTO Departments (DepartmentID, DepartmentName)
VALUES (1, 'HR'),
       (2, 'IT');

-- Using LEFT JOIN to get all employees and their department names
SELECT Employees.FirstName, Employees.LastName, Departments.DepartmentName
FROM Employees
LEFT JOIN Departments
ON Employees.DepartmentID = Departments.DepartmentID;

```

**Expected Output:**

```
| FirstName | LastName | DepartmentName |
|-----------|----------|----------------|
| John      | Doe      | HR             |
| Jane      | Smith    | IT             |
| James     | Bond     | NULL           |

```

---

### **RIGHT JOIN**

A `RIGHT JOIN` returns all records from the right table (Departments), along with matching records from the left table (Employees). If there’s no match, NULL values will be returned for the left table.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT
);

-- Creating the Departments table
CREATE TABLE Departments (
    DepartmentID INT,
    DepartmentName VARCHAR(50)
);

-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID)
VALUES (1, 'John', 'Doe', 1),
       (2, 'Jane', 'Smith', 2);

-- Inserting data into Departments table
INSERT INTO Departments (DepartmentID, DepartmentName)
VALUES (1, 'HR'),
       (2, 'IT'),
       (3, 'Finance');

-- Using RIGHT JOIN to get all departments and their employees
SELECT Employees.FirstName, Employees.LastName, Departments.DepartmentName
FROM Employees
RIGHT JOIN Departments
ON Employees.DepartmentID = Departments.DepartmentID;

```

**Expected Output:**

```
| FirstName | LastName | DepartmentName |
|-----------|----------|----------------|
| John      | Doe      | HR             |
| Jane      | Smith    | IT             |
| NULL      | NULL     | Finance        |

```

---

### **FULL JOIN**

A `FULL JOIN` returns all records when there’s a match in either left (Employees) or right (Departments) table. If there’s no match, NULL values are returned for the missing side.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT
);

-- Creating the Departments table
CREATE TABLE Departments (
    DepartmentID INT,
    DepartmentName VARCHAR(50)
);

-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID)
VALUES (1, 'John', 'Doe', 1),
       (2, 'Jane', 'Smith', 2),
       (3, 'James', 'Bond', NULL);

-- Inserting data into Departments table
INSERT INTO Departments (DepartmentID, DepartmentName)
VALUES (1, 'HR'),
       (2, 'IT'),
       (3, 'Finance'),
       (4, 'Marketing');

-- Using FULL JOIN to get all employees and all departments
SELECT Employees.FirstName, Employees.LastName, Departments.DepartmentName
FROM Employees
FULL JOIN Departments
ON Employees.DepartmentID = Departments.DepartmentID;

```

**Expected Output:**

```
| FirstName | LastName | DepartmentName |
|-----------|----------|----------------|
| John      | Doe      | HR             |
| Jane      | Smith    | IT             |
| James     | Bond     | NULL           |
| NULL      | NULL     | Finance        |
| NULL      | NULL     | Marketing      |

```

---

### **UNION**

The `UNION` operator combines the results of two or more `SELECT` statements and removes duplicates. The number and type of columns in each `SELECT` statement must be the same.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50)
);

-- Creating the Contractors table
CREATE TABLE Contractors (
    ContractorID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50)
);

-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName)
VALUES (1, 'John', 'Doe'),
       (2, 'Jane', 'Smith');

-- Inserting data into Contractors table
INSERT INTO Contractors (ContractorID, FirstName, LastName)
VALUES (1, 'James', 'Bond'),
       (2, 'Alice', 'Johnson');

-- Using UNION to combine Employees and Contractors (without duplicates)
SELECT FirstName, LastName FROM Employees
UNION
SELECT FirstName, LastName FROM Contractors;

```

**Expected Output:**

```
| FirstName | LastName |
|-----------|----------|
| John      | Doe      |
| Jane      | Smith    |
| James     | Bond     |
| Alice     | Johnson  |

```

---

### **UNION ALL**

The `UNION ALL` operator combines the results of two or more `SELECT` statements and includes all duplicates.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50)
);

-- Creating the Contractors table
CREATE TABLE Contractors (
    ContractorID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50)
);

-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName)
VALUES (1, 'John', 'Doe'),
       (2, 'Jane', 'Smith');

-- Inserting data into Contractors table
INSERT INTO Contractors (ContractorID, FirstName, LastName)
VALUES (1, 'James', 'Bond'),
       (2, 'Alice', 'Johnson'),
       (1, 'James', 'Bond');  -- Duplicate entry

-- Using UNION ALL to combine Employees and Contractors (including duplicates)
SELECT FirstName, LastName FROM Employees
UNION ALL
SELECT FirstName, LastName FROM Contractors;

```

**Expected Output:**

```
| FirstName | LastName |
|-----------|----------|
| John      | Doe      |
| Jane      | Smith    |
| James     | Bond     |
| Alice     | Johnson  |
| James     | Bond     |

```

---

These examples cover the most common types of joins (`INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL JOIN`) and the union operations (`UNION` and `UNION ALL`). Each example demonstrates how to join tables based on common columns and combine query results while handling duplicates accordingly.

## **Subqueries**

### **Creating a Table**

Let's use the **Employees** and **Departments** tables from the previous example to demonstrate subqueries.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT,
    Salary INT
);

-- Creating the Departments table
CREATE TABLE Departments (
    DepartmentID INT,
    DepartmentName VARCHAR(50)
);

```

---

### **Subquery in SELECT**

A **Subquery in SELECT** allows you to return a value derived from a subquery in the main query's `SELECT` list. The subquery must return a single value for each row processed by the main query.

```sql
-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (1, 'John', 'Doe', 1, 50000),
       (2, 'Jane', 'Smith', 2, 60000),
       (3, 'James', 'Bond', 3, 70000);

-- Inserting data into Departments table
INSERT INTO Departments (DepartmentID, DepartmentName)
VALUES (1, 'HR'),
       (2, 'IT'),
       (3, 'Finance');

-- Using a Subquery in SELECT to get the average salary for each employee's department
SELECT
    FirstName,
    LastName,
    Salary,
    (SELECT AVG(Salary) FROM Employees e WHERE e.DepartmentID = Employees.DepartmentID) AS AvgDepartmentSalary
FROM Employees;

```

**Expected Output:**

```
| FirstName | LastName | Salary | AvgDepartmentSalary |
|-----------|----------|--------|---------------------|
| John      | Doe      | 50000  | 50000               |
| Jane      | Smith    | 60000  | 60000               |
| James     | Bond     | 70000  | 70000               |

```

In this example, the subquery computes the average salary for each employee’s department and is included in the main query’s `SELECT` list. For each row, it returns the average salary of the department where the employee works.

---

### **Subquery in WHERE**

A **Subquery in WHERE** is used to filter results based on the results of another query. The subquery returns a set of values that are then compared in the main query’s `WHERE` clause.

```sql
-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (1, 'John', 'Doe', 1, 50000),
       (2, 'Jane', 'Smith', 2, 60000),
       (3, 'James', 'Bond', 3, 70000),
       (4, 'Alice', 'Johnson', 2, 65000);

-- Using a Subquery in WHERE to find employees whose salary is higher than the average salary in their department
SELECT FirstName, LastName, Salary
FROM Employees
WHERE Salary > (SELECT AVG(Salary) FROM Employees e WHERE e.DepartmentID = Employees.DepartmentID);

```

**Expected Output:**

```
| FirstName | LastName | Salary |
|-----------|----------|--------|
| Jane      | Smith    | 60000  |
| Alice     | Johnson  | 65000  |

```

In this example, the subquery calculates the average salary for each department, and the main query returns employees whose salary is greater than the average salary for their respective departments.

---

### **Subquery in SELECT with IN**

You can also use a subquery with the `IN` operator to match a column value to a list of values returned by the subquery.

```sql
-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (1, 'John', 'Doe', 1, 50000),
       (2, 'Jane', 'Smith', 2, 60000),
       (3, 'James', 'Bond', 3, 70000),
       (4, 'Alice', 'Johnson', 2, 65000);

-- Using a Subquery in WHERE with IN to get employees who work in departments that have a salary greater than 60000
SELECT FirstName, LastName, DepartmentID
FROM Employees
WHERE DepartmentID IN (SELECT DepartmentID FROM Employees WHERE Salary > 60000);

```

**Expected Output:**

```
| FirstName | LastName | DepartmentID |
|-----------|----------|--------------|
| Jane      | Smith    | 2            |
| Alice     | Johnson  | 2            |
| James     | Bond     | 3            |

```

Here, the subquery finds departments where the average salary is greater than 60000, and the main query returns employees working in those departments.

---

### **Subquery with EXISTS**

The `EXISTS` operator is used to check whether the result of the subquery returns any rows. If the subquery returns at least one row, the condition evaluates to `TRUE`.

```sql
-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (1, 'John', 'Doe', 1, 50000),
       (2, 'Jane', 'Smith', 2, 60000),
       (3, 'James', 'Bond', 3, 70000),
       (4, 'Alice', 'Johnson', 2, 65000);

-- Using a Subquery with EXISTS to find employees who work in departments that have employees with salaries greater than 60000
SELECT FirstName, LastName
FROM Employees e
WHERE EXISTS (SELECT 1 FROM Employees WHERE DepartmentID = e.DepartmentID AND Salary > 60000);

```

**Expected Output:**

```
| FirstName | LastName |
|-----------|----------|
| Jane      | Smith    |
| Alice     | Johnson  |
| James     | Bond     |

```

Here, the subquery checks for each department whether there are any employees whose salary is greater than 60000. The main query then returns employees who work in these departments.

---

### **Subquery with Correlation (Self Join)**

A **correlated subquery** references a column from the outer query. Each row processed by the outer query can result in a different subquery execution.

```sql
-- Inserting data into Employees table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (1, 'John', 'Doe', 1, 50000),
       (2, 'Jane', 'Smith', 2, 60000),
       (3, 'James', 'Bond', 3, 70000),
       (4, 'Alice', 'Johnson', 2, 65000);

-- Using a correlated subquery to find employees with a higher salary than the average salary in their department
SELECT FirstName, LastName, Salary
FROM Employees e1
WHERE Salary > (SELECT AVG(Salary) FROM Employees e2 WHERE e2.DepartmentID = e1.DepartmentID);

```

**Expected Output:**

```
| FirstName | LastName | Salary |
|-----------|----------|--------|
| Jane      | Smith    | 60000  |
| Alice     | Johnson  | 65000  |

```

In this case, the subquery is correlated with the outer query because it references `e1.DepartmentID`. It calculates the average salary for each department and compares the salary of each employee to the average of their department.

---

These examples cover the use of **subqueries** in both `SELECT` and `WHERE` clauses. Subqueries allow you to perform complex queries that depend on values or conditions calculated from another query.

## **Stored Procedures**

### **Stored Procedures**

A **Stored Procedure** is a set of SQL statements that you can save and reuse. It allows you to encapsulate logic into a single callable unit, making your SQL code cleaner and more modular. Stored procedures can accept parameters and return values.

---

### **Creating a Stored Procedure**

Let's create a stored procedure for inserting a new employee into the **Employees** table.

```sql
-- Creating the Employees table (if it doesn't exist)
CREATE TABLE Employees (
    EmployeeID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT,
    Salary INT
);

-- Creating a Stored Procedure for inserting a new employee
DELIMITER $$

CREATE PROCEDURE InsertEmployee(
    IN p_EmployeeID INT,
    IN p_FirstName VARCHAR(50),
    IN p_LastName VARCHAR(50),
    IN p_DepartmentID INT,
    IN p_Salary INT
)
BEGIN
    INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
    VALUES (p_EmployeeID, p_FirstName, p_LastName, p_DepartmentID, p_Salary);
END $$

DELIMITER ;

```

In this example:

- We create a procedure called `InsertEmployee` that takes five parameters: `p_EmployeeID`, `p_FirstName`, `p_LastName`, `p_DepartmentID`, and `p_Salary`.
- The procedure inserts these values into the **Employees** table.

---

### **Calling a Stored Procedure**

Once the stored procedure is created, you can call it to insert data into the **Employees** table.

```sql
-- Calling the Stored Procedure to insert a new employee
CALL InsertEmployee(5, 'Alice', 'Johnson', 2, 65000);

```

**Expected Output:**

```
-- No direct output, but the data is inserted into the Employees table.

```

This will insert a new employee with `EmployeeID = 5`, `FirstName = 'Alice'`, `LastName = 'Johnson'`, `DepartmentID = 2`, and `Salary = 65000`.

---

### **Stored Procedure with Output**

You can also create stored procedures that return values, like calculating the average salary for a given department.

```sql
-- Creating a Stored Procedure to calculate the average salary in a department
DELIMITER $$

CREATE PROCEDURE GetAvgSalaryByDepartment(
    IN p_DepartmentID INT,
    OUT p_AvgSalary DECIMAL(10,2)
)
BEGIN
    SELECT AVG(Salary) INTO p_AvgSalary
    FROM Employees
    WHERE DepartmentID = p_DepartmentID;
END $$

DELIMITER ;

```

### **Calling the Stored Procedure with Output**

```sql
-- Declaring a variable to store the result
DECLARE @AvgSalary DECIMAL(10,2);

-- Calling the Stored Procedure and getting the average salary for department 2
CALL GetAvgSalaryByDepartment(2, @AvgSalary);

-- Displaying the result
SELECT @AvgSalary AS AvgSalaryForDepartment2;

```

**Expected Output:**

```
| AvgSalaryForDepartment2 |
|-------------------------|
| 62500.00               |

```

---

## **Triggers**

A **Trigger** is a set of SQL statements that automatically executes or “fires” when a specified event occurs on a particular table or view. Triggers are used for tasks such as enforcing business rules, updating other tables, or logging activities.

---

### **Creating a Trigger**

Let's create a **trigger** that automatically updates an employee's `LastUpdated` timestamp whenever their salary is updated.

```sql
-- Creating the Employees table (if it doesn't exist)
CREATE TABLE Employees (
    EmployeeID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT,
    Salary INT,
    LastUpdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Creating a Trigger to update the LastUpdated timestamp when the salary is updated
DELIMITER $$

CREATE TRIGGER UpdateSalaryTimestamp
AFTER UPDATE ON Employees
FOR EACH ROW
BEGIN
    IF OLD.Salary <> NEW.Salary THEN
        UPDATE Employees
        SET LastUpdated = CURRENT_TIMESTAMP
        WHERE EmployeeID = NEW.EmployeeID;
    END IF;
END $$

DELIMITER ;

```

In this example:

- The trigger `UpdateSalaryTimestamp` is defined to execute **after an update** on the **Employees** table.
- The trigger updates the `LastUpdated` timestamp whenever the `Salary` column is modified.

---

### **Trigger Execution Example**

```sql
-- Inserting an employee into the table
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (1, 'John', 'Doe', 1, 50000);

-- Updating the salary of the employee, which will fire the trigger
UPDATE Employees
SET Salary = 55000
WHERE EmployeeID = 1;

```

**Expected Output:**

The **LastUpdated** timestamp of the employee with `EmployeeID = 1` will be updated automatically to the current timestamp. You can verify it by selecting the row.

```sql
SELECT * FROM Employees WHERE EmployeeID = 1;

```

Output:

```
| EmployeeID | FirstName | LastName | DepartmentID | Salary | LastUpdated          |
|------------|-----------|----------|--------------|--------|----------------------|
| 1          | John      | Doe      | 1            | 55000  | 2025-01-27 14:30:00  |

```

---

### **Trigger with BEFORE and AFTER**

You can define triggers that fire either **before** or **after** a modification event occurs.

```sql
-- Creating a BEFORE INSERT trigger that ensures no employee can be inserted with a negative salary
DELIMITER $$

CREATE TRIGGER BeforeInsertEmployee
BEFORE INSERT ON Employees
FOR EACH ROW
BEGIN
    IF NEW.Salary < 0 THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Salary cannot be negative';
    END IF;
END $$

DELIMITER ;

```

In this example:

- The trigger `BeforeInsertEmployee` fires **before** a new employee is inserted into the **Employees** table.
- If the `Salary` value is negative, the trigger prevents the insertion and raises an error.

---

### **Trigger Execution Example with BEFORE INSERT**

```sql
-- Attempting to insert an employee with a negative salary
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (2, 'Jane', 'Smith', 2, -50000);

```

**Expected Output:**

```
ERROR 1644 (45000): Salary cannot be negative

```

This ensures that no employee is added with a negative salary.

---

### **Trigger Example for Logging**

Let's create a trigger that logs salary changes to a separate **SalaryHistory** table.

```sql
-- Creating the SalaryHistory table
CREATE TABLE SalaryHistory (
    EmployeeID INT,
    OldSalary INT,
    NewSalary INT,
    ChangeDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Creating the Trigger to log salary changes
DELIMITER $$

CREATE TRIGGER LogSalaryChange
AFTER UPDATE ON Employees
FOR EACH ROW
BEGIN
    IF OLD.Salary <> NEW.Salary THEN
        INSERT INTO SalaryHistory (EmployeeID, OldSalary, NewSalary)
        VALUES (NEW.EmployeeID, OLD.Salary, NEW.Salary);
    END IF;
END $$

DELIMITER ;

```

Now, whenever the salary of an employee is updated, the **SalaryHistory** table will automatically record the change.

---

### **Trigger Execution Example for Logging**

```sql
-- Updating the salary to trigger the logging
UPDATE Employees
SET Salary = 60000
WHERE EmployeeID = 1;

```

**Expected Output:**

The **SalaryHistory** table will have the following row added:

```
| EmployeeID | OldSalary | NewSalary | ChangeDate           |
|------------|-----------|-----------|----------------------|
| 1          | 55000     | 60000     | 2025-01-27 14:45:00  |

```

---

These examples cover **Stored Procedures** and **Triggers**. Stored procedures allow you to encapsulate logic for reuse, while triggers enable automatic responses to data changes. Both are powerful tools for managing data in SQL databases.

## **Transactions**

### **Transactions**

A **Transaction** in SQL is a sequence of operations performed as a single unit. Transactions ensure data integrity and consistency, allowing multiple operations to either fully complete or completely roll back in case of errors. Transactions are typically used to group multiple related database operations (such as multiple insertions or updates) into one atomic unit, ensuring that the database remains consistent even in case of failures.

### **ACID Properties of a Transaction:**

1. **Atomicity**: All operations in the transaction are completed or none.
2. **Consistency**: The database must always be in a consistent state before and after the transaction.
3. **Isolation**: Transactions are isolated from each other, and intermediate results are not visible to other transactions.
4. **Durability**: Once a transaction is committed, its changes are permanent.

---

### **Using Transactions in SQL**

Here’s how to use transactions in SQL:

```sql
-- Start the transaction
START TRANSACTION;

-- Example of multiple SQL operations within a transaction
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (6, 'Sam', 'Brown', 3, 75000);

UPDATE Employees
SET Salary = Salary + 5000
WHERE DepartmentID = 3;

-- Commit the transaction to save changes
COMMIT;

```

In this example:

- A new employee is inserted.
- Salaries of all employees in department 3 are updated.
- Finally, the transaction is committed to make the changes permanent.

---

### **Rolling Back a Transaction**

If there’s an error or you decide not to apply the changes, you can roll back the transaction, which undoes all the operations done so far.

```sql
-- Start the transaction
START TRANSACTION;

-- Example of a transaction
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (7, 'Eve', 'Davis', 4, 68000);

-- Simulating an error: Let's say a constraint is violated
UPDATE Employees
SET Salary = -2000  -- This could be an invalid operation due to a constraint
WHERE EmployeeID = 7;

-- Roll back the transaction to undo changes
ROLLBACK;

```

**Expected Output:**

- Since the `UPDATE` operation causes an error (possibly due to a constraint violation), the transaction is rolled back, and the inserted employee data is not saved.

---

### **Savepoints in Transactions**

You can set a **savepoint** in a transaction, which allows you to roll back to a specific point within the transaction, instead of rolling back all changes.

```sql
-- Start the transaction
START TRANSACTION;

-- Insert a new employee
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (8, 'Tom', 'Wilson', 2, 72000);

-- Create a savepoint
SAVEPOINT Savepoint1;

-- Simulate an error or decide to undo the changes made after the savepoint
UPDATE Employees
SET Salary = 78000
WHERE EmployeeID = 8;

-- Rolling back to the savepoint
ROLLBACK TO SAVEPOINT Savepoint1;

-- Commit the transaction after rolling back to savepoint
COMMIT;

```

**Expected Output:**

- The employee's salary update is rolled back, but the initial insert operation is retained and committed.

---

## **Indexing**

**Indexing** is a performance optimization technique in databases that improves the speed of data retrieval operations. An index creates a data structure (such as a B-tree) to speed up searches. It is used to quickly locate data without having to search every row in a table.

---

### **Creating an Index**

Let’s create an index on the **Salary** column in the **Employees** table to speed up searches based on salary.

```sql
-- Creating an index on the Salary column
CREATE INDEX idx_salary ON Employees (Salary);

```

This index helps speed up queries that filter or sort data based on the **Salary** column.

---

### **Using Indexes in Queries**

Once the index is created, you can use it implicitly when running queries. For example, let’s say we want to find employees with a salary greater than 70,000:

```sql
-- Query that benefits from the index on Salary
SELECT * FROM Employees
WHERE Salary > 70000;

```

The query will be executed faster because the database uses the index to locate employees with high salaries rather than scanning the entire table.

---

### **Unique Indexes**

A **Unique Index** ensures that all values in the indexed column(s) are unique. It's commonly used for enforcing uniqueness on non-primary key columns.

```sql
-- Creating a unique index on the Employee's LastName to prevent duplicate entries
CREATE UNIQUE INDEX idx_lastname ON Employees (LastName);

```

This ensures that no two employees can have the same **LastName** in the table.

---

### **Composite Indexes**

A **Composite Index** is an index on multiple columns. It’s useful when you frequently query the table with multiple conditions (on different columns).

```sql
-- Creating a composite index on both DepartmentID and Salary columns
CREATE INDEX idx_dept_salary ON Employees (DepartmentID, Salary);

```

This index will be used for queries that filter by both **DepartmentID** and **Salary**.

---

### **Dropping an Index**

If an index is no longer needed or has become inefficient, you can drop it to save resources.

```sql
-- Dropping an index
DROP INDEX idx_salary ON Employees;

```

This removes the **idx_salary** index from the **Employees** table.

---

### **Clustered vs. Non-Clustered Indexes**

- **Clustered Index**: The data rows are stored in the same order as the index. A table can have only one clustered index, which is typically the **primary key**.
- **Non-Clustered Index**: The data rows are stored separately from the index. You can have multiple non-clustered indexes on a table.

### **Example of Creating a Clustered Index**

```sql
-- Creating a clustered index on the EmployeeID column
CREATE CLUSTERED INDEX idx_employeeid ON Employees (EmployeeID);

```

---

### **Full-Text Indexing**

For text-heavy queries, such as searching for words or phrases in long text fields, you can use **Full-Text Indexing**.

```sql
-- Creating a full-text index on the Notes column
CREATE FULLTEXT INDEX idx_notes ON Employees (Notes);

```

This index type speeds up text searches using conditions like `MATCH ... AGAINST` for natural language searches.

---

### **Query Optimization with Indexing**

Indexes improve query performance by reducing the number of rows that need to be scanned, especially for large tables. However, creating too many indexes or indexing the wrong columns can slow down **INSERT**, **UPDATE**, and **DELETE** operations because the indexes need to be maintained.

---

## **Views in SQL**

### **What is a View?**

A **View** in SQL is a virtual table that represents the result of a query. It doesn’t store data itself, but provides a way to structure and simplify complex queries. When you query a view, the database engine dynamically retrieves the data based on the underlying tables.

Views can simplify complex queries, encapsulate business logic, and provide a consistent way to access data without exposing the underlying tables directly.

---

### **Creating a View**

To create a view, you use the `CREATE VIEW` statement followed by the name of the view and the query that defines it.

```sql
-- Creating a View for Employees' Full Information
CREATE VIEW EmployeeInfo AS
SELECT EmployeeID, FirstName, LastName, Age, Salary
FROM Employees;

```

In this example:

- A view named `EmployeeInfo` is created, which shows the `EmployeeID`, `FirstName`, `LastName`, `Age`, and `Salary` columns from the **Employees** table.

---

### **Querying a View**

Once a view is created, you can query it just like a regular table using a `SELECT` statement.

```sql
-- Querying the EmployeeInfo view to retrieve employee details
SELECT * FROM EmployeeInfo;

```

This query retrieves all the columns from the `EmployeeInfo` view, which in turn pulls the data from the **Employees** table.

**Expected Output:**

```
EmployeeID | FirstName | LastName | Age | Salary
------------------------------------------------
1          | John      | Doe      | 30  | 55000
2          | Jane      | Smith    | 25  | 48000
3          | Bob       | Johnson  | 40  | 60000
...

```

---

### **Updating a View**

In SQL, you can update a view’s underlying data if the view is based on a single table and does not involve complex operations like joins or aggregations. The changes reflect in the base table that the view is derived from.

```sql
-- Updating the salary of an employee through the EmployeeInfo view
UPDATE EmployeeInfo
SET Salary = 60000
WHERE EmployeeID = 2;

```

This query updates the salary of the employee with `EmployeeID = 2` in the **Employees** table.

**Expected Output:**

- The employee with `EmployeeID = 2` will now have a salary of 60,000 in the **Employees** table.

---

### **Dropping a View**

To remove a view, you can use the `DROP VIEW` statement. This operation only removes the view definition and doesn’t affect the underlying table data.

```sql
-- Dropping the EmployeeInfo view
DROP VIEW EmployeeInfo;

```

After executing this statement, the `EmployeeInfo` view is removed, and you can no longer query it.

---

### **Use Cases for Views**

**1. Simplify Complex Queries:**

- Views can encapsulate complex joins or calculations, allowing users to query the view rather than writing long, complicated queries every time.

```sql
-- A complex query that joins multiple tables and calculates salary
CREATE VIEW EmployeeDepartmentInfo AS
SELECT e.EmployeeID, e.FirstName, e.LastName, e.Salary, d.DepartmentName
FROM Employees e
JOIN Departments d ON e.DepartmentID = d.DepartmentID;

```

**2. Provide Security and Abstraction:**

- Views can be used to restrict access to sensitive data. For example, you can create a view that only returns certain columns, such as excluding the `Salary` column, to ensure privacy.

```sql
-- Creating a view that hides sensitive information (Salary)
CREATE VIEW EmployeePublicInfo AS
SELECT EmployeeID, FirstName, LastName, Age
FROM Employees;

```

**3. Data Aggregation:**

- Views can be used to store the results of data aggregation, like summing or averaging values, to simplify reports.

```sql
-- A view that calculates the average salary by department
CREATE VIEW AvgSalaryByDepartment AS
SELECT DepartmentID, AVG(Salary) AS AvgSalary
FROM Employees
GROUP BY DepartmentID;

```

**4. Reporting:**

- Views are often used for reporting purposes, where you need a consistent way to retrieve data in a specific format for end-users or business intelligence tools.

```sql
-- A report view to show employee names and their department names
CREATE VIEW EmployeeReport AS
SELECT e.FirstName, e.LastName, d.DepartmentName
FROM Employees e
JOIN Departments d ON e.DepartmentID = d.DepartmentID;

```

**5. Data Transformation:**

- Views can be used to transform the data, such as converting currency values or adjusting dates, to match the format needed for business purposes.

```sql
-- Creating a view to transform employee salary to a different currency (e.g., USD to EUR)
CREATE VIEW EmployeeSalaryInEuro AS
SELECT EmployeeID, FirstName, LastName, Salary * 0.85 AS SalaryInEuro
FROM Employees;

```

---

### **Limitations of Views**

- Views cannot always be updated if they involve multiple tables, aggregations, or joins.
- They do not store data, so every time you query a view, it recalculates the result.
- They might not be as performant as querying the underlying tables directly, especially for complex views.

---

## **Window Functions**

### **ROW_NUMBER(), RANK(), DENSE_RANK()**

Window functions are used to perform calculations across a set of rows related to the current row, often without collapsing the result set like aggregation functions do. Here, we'll focus on `ROW_NUMBER()`, `RANK()`, and `DENSE_RANK()`, which assign a ranking to rows within a partition of a result set.

### **ROW_NUMBER()**

`ROW_NUMBER()` assigns a unique number to each row in the result set. The numbering is done sequentially starting from 1. It's useful for assigning a unique identifier to rows.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting sample data
INSERT INTO Employees (EmployeeID, FirstName, LastName, Age, Salary)
VALUES (1, 'John', 'Doe', 30, 55000),
       (2, 'Jane', 'Smith', 25, 48000),
       (3, 'Alice', 'Johnson', 35, 60000),
       (4, 'Bob', 'Williams', 40, 65000),
       (5, 'Charlie', 'Brown', 28, 47000);

-- Using ROW_NUMBER() to assign a unique number to each employee based on Salary, ordered in descending order
SELECT EmployeeID, FirstName, LastName, Salary,
       ROW_NUMBER() OVER (ORDER BY Salary DESC) AS RowNumber
FROM Employees;

```

**Expected Output:**

```
EmployeeID | FirstName | LastName | Salary | RowNumber
------------------------------------------------------
4          | Bob       | Williams | 65000  | 1
3          | Alice     | Johnson  | 60000  | 2
1          | John      | Doe      | 55000  | 3
2          | Jane      | Smith    | 48000  | 4
5          | Charlie   | Brown    | 47000  | 5

```

### **RANK()**

`RANK()` assigns a rank to each row within the partition of a result set. If two rows have the same value, they will receive the same rank, but the next row will have a rank that skips the number(s) based on the number of tied rows.

```sql
-- Using RANK() to rank employees based on Salary, ordered in descending order
SELECT EmployeeID, FirstName, LastName, Salary,
       RANK() OVER (ORDER BY Salary DESC) AS Rank
FROM Employees;

```

**Expected Output:**

```
EmployeeID | FirstName | LastName | Salary | Rank
---------------------------------------------------
4          | Bob       | Williams | 65000  | 1
3          | Alice     | Johnson  | 60000  | 2
1          | John      | Doe      | 55000  | 3
2          | Jane      | Smith    | 48000  | 4
5          | Charlie   | Brown    | 47000  | 5

```

### **DENSE_RANK()**

`DENSE_RANK()` also assigns ranks to rows, but unlike `RANK()`, it does not skip any rank numbers when there are ties.

```sql
-- Using DENSE_RANK() to rank employees based on Salary, ordered in descending order
SELECT EmployeeID, FirstName, LastName, Salary,
       DENSE_RANK() OVER (ORDER BY Salary DESC) AS DenseRank
FROM Employees;

```

**Expected Output:**

```
EmployeeID | FirstName | LastName | Salary | DenseRank
------------------------------------------------------
4          | Bob       | Williams | 65000  | 1
3          | Alice     | Johnson  | 60000  | 2
1          | John      | Doe      | 55000  | 3
2          | Jane      | Smith    | 48000  | 4
5          | Charlie   | Brown    | 47000  | 5

```

---

### **PARTITION BY**

The `PARTITION BY` clause is used in conjunction with window functions to divide the result set into partitions and perform calculations within each partition separately. This is useful when you want to apply window functions over different groups of data.

```sql
-- Creating the Departments table
CREATE TABLE Departments (
    DepartmentID INT,
    DepartmentName VARCHAR(50),
    EmployeeID INT
);

-- Inserting sample data
INSERT INTO Departments (DepartmentID, DepartmentName, EmployeeID)
VALUES (1, 'HR', 1),
       (2, 'Finance', 2),
       (3, 'Engineering', 3),
       (1, 'HR', 4),
       (2, 'Finance', 5);

-- Using PARTITION BY to rank employees within their departments based on Salary
SELECT e.EmployeeID, e.FirstName, e.LastName, e.Salary, d.DepartmentName,
       RANK() OVER (PARTITION BY d.DepartmentID ORDER BY e.Salary DESC) AS DepartmentRank
FROM Employees e
JOIN Departments d ON e.EmployeeID = d.EmployeeID;

```

**Expected Output:**

```
EmployeeID | FirstName | LastName | Salary | DepartmentName | DepartmentRank
----------------------------------------------------------------------
1          | John      | Doe      | 55000  | HR             | 1
4          | Bob       | Williams | 65000  | HR             | 1
2          | Jane      | Smith    | 48000  | Finance        | 1
5          | Charlie   | Brown    | 47000  | Finance        | 2
3          | Alice     | Johnson  | 60000  | Engineering    | 1

```

In this example, the ranking is done within each department (`PARTITION BY DepartmentID`).

---

### **Aggregate with OVER()**

Using an aggregate function like `SUM()`, `AVG()`, `MIN()`, or `MAX()` with `OVER()` allows you to calculate aggregate values without collapsing the result set. The `OVER()` clause is used to specify the window over which the aggregate function should operate.

### **SUM() with OVER()**

```sql
-- Using SUM() with OVER() to calculate the total salary for all employees, without collapsing the result set
SELECT EmployeeID, FirstName, LastName, Salary,
       SUM(Salary) OVER () AS TotalSalary
FROM Employees;

```

**Expected Output:**

```
EmployeeID | FirstName | LastName | Salary | TotalSalary
-------------------------------------------------------
1          | John      | Doe      | 55000  | 285000
2          | Jane      | Smith    | 48000  | 285000
3          | Alice     | Johnson  | 60000  | 285000
4          | Bob       | Williams | 65000  | 285000
5          | Charlie   | Brown    | 47000  | 285000

```

Here, the total salary (`TotalSalary`) is calculated across all employees, but each row still contains individual employee data.

### **AVG() with OVER()**

```sql
-- Using AVG() with OVER() to calculate the average salary of all employees, without collapsing the result set
SELECT EmployeeID, FirstName, LastName, Salary,
       AVG(Salary) OVER () AS AverageSalary
FROM Employees;

```

**Expected Output:**

```
EmployeeID | FirstName | LastName | Salary | AverageSalary
--------------------------------------------------------
1          | John      | Doe      | 55000  | 57000
2          | Jane      | Smith    | 48000  | 57000
3          | Alice     | Johnson  | 60000  | 57000
4          | Bob       | Williams | 65000  | 57000
5          | Charlie   | Brown    | 47000  | 57000

```

In this case, the average salary (`AverageSalary`) is calculated over all employees, but the result still retains individual employee data.

## **Common Table Expressions (CTEs)**

Common Table Expressions (CTEs) provide a way to create temporary result sets that can be referenced within a `SELECT`, `INSERT`, `UPDATE`, or `DELETE` statement. CTEs are often used to simplify complex queries and improve readability. They are defined using the `WITH` clause.

### **WITH Clauses**

The `WITH` clause allows you to define one or more CTEs that can be used in the main query. CTEs can be thought of as temporary tables or views that are valid only within the scope of the query.

```sql
-- Creating the Employees table
CREATE TABLE Employees (
    EmployeeID INT,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT,
    Salary DECIMAL(10, 2)
);

-- Inserting sample data
INSERT INTO Employees (EmployeeID, FirstName, LastName, DepartmentID, Salary)
VALUES (1, 'John', 'Doe', 1, 55000),
       (2, 'Jane', 'Smith', 1, 48000),
       (3, 'Alice', 'Johnson', 2, 60000),
       (4, 'Bob', 'Williams', 3, 65000),
       (5, 'Charlie', 'Brown', 2, 47000);

-- Using WITH clause to define a CTE
WITH DepartmentSalaries AS (
    SELECT DepartmentID, AVG(Salary) AS AverageSalary
    FROM Employees
    GROUP BY DepartmentID
)
-- Using the CTE in the main query
SELECT e.EmployeeID, e.FirstName, e.LastName, e.Salary, ds.AverageSalary
FROM Employees e
JOIN DepartmentSalaries ds ON e.DepartmentID = ds.DepartmentID;

```

**Expected Output:**

```
EmployeeID | FirstName | LastName | Salary | AverageSalary
---------------------------------------------------------
1          | John      | Doe      | 55000  | 51000
2          | Jane      | Smith    | 48000  | 51000
3          | Alice     | Johnson  | 60000  | 53500
4          | Bob       | Williams | 65000  | 53500
5          | Charlie   | Brown    | 47000  | 53500

```

In this example, `DepartmentSalaries` is a CTE that calculates the average salary for each department. The main query joins the CTE with the `Employees` table to show each employee's salary along with their department's average salary.

CTEs can be recursive, allowing queries to reference themselves, which is useful for hierarchical data (e.g., organizational charts).

---

## **Data Manipulation with CASE**

The `CASE` expression provides conditional logic within SQL queries. It works like an `IF` statement in other programming languages and allows you to return specific values based on conditions. It can be used in `SELECT`, `INSERT`, `UPDATE`, and `DELETE` statements.

### **Conditional Logic**

The `CASE` expression has two forms:

1. **Simple CASE**: Compares an expression to multiple values.
2. **Searched CASE**: Evaluates each condition independently.

### **Simple CASE**

```sql
-- Using CASE in a SELECT statement to categorize employees based on salary
SELECT EmployeeID, FirstName, LastName, Salary,
       CASE Salary
           WHEN 65000 THEN 'High Salary'
           WHEN 60000 THEN 'Medium Salary'
           ELSE 'Low Salary'
       END AS SalaryCategory
FROM Employees;

```

**Expected Output:**

```
EmployeeID | FirstName | LastName | Salary | SalaryCategory
--------------------------------------------------------
1          | John      | Doe      | 55000  | Low Salary
2          | Jane      | Smith    | 48000  | Low Salary
3          | Alice     | Johnson  | 60000  | Medium Salary
4          | Bob       | Williams | 65000  | High Salary
5          | Charlie   | Brown    | 47000  | Low Salary

```

In this example, the `CASE` expression is used to categorize employees based on their salary. If the salary is `65000`, the employee is classified as having a "High Salary," if `60000`, "Medium Salary," otherwise "Low Salary."

### **Searched CASE**

```sql
-- Using Searched CASE for more complex conditions
SELECT EmployeeID, FirstName, LastName, Salary,
       CASE
           WHEN Salary >= 60000 THEN 'High Salary'
           WHEN Salary >= 50000 THEN 'Medium Salary'
           ELSE 'Low Salary'
       END AS SalaryCategory
FROM Employees;

```

**Expected Output:**

```
EmployeeID | FirstName | LastName | Salary | SalaryCategory
--------------------------------------------------------
1          | John      | Doe      | 55000  | Medium Salary
2          | Jane      | Smith    | 48000  | Low Salary
3          | Alice     | Johnson  | 60000  | High Salary
4          | Bob       | Williams | 65000  | High Salary
5          | Charlie   | Brown    | 47000  | Low Salary

```

In the searched version of `CASE`, each condition is evaluated in order, and the corresponding result is returned. If an employee's salary is `>= 60000`, they are categorized as "High Salary"; if `>= 50000`, "Medium Salary"; otherwise, "Low Salary."