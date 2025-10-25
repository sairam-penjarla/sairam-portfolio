## Easy SQL Questions

### **Basics & Concepts**

1. **What is SQL?**

   * SQL (Structured Query Language) is used to **manage and manipulate relational databases**.

2. **Difference between SQL and MySQL**

   * SQL → language
   * MySQL → database management system using SQL

3. **What are tables in SQL?**

   * Tables store data in **rows and columns**.

4. **What is a primary key?**

   * A column (or combination) that **uniquely identifies** each row in a table.

5. **What is a foreign key?**

   * A column in one table that refers to the **primary key of another table**, establishing a relationship.

6. **Difference between `CHAR` and `VARCHAR`**

   * `CHAR` → fixed length
   * `VARCHAR` → variable length

7. **Difference between `WHERE` and `HAVING`**

   * `WHERE` → filters **rows before aggregation**
   * `HAVING` → filters **groups after aggregation**

8. **What is `NULL` in SQL?**

   * Represents **unknown or missing values**.

9. **Difference between `DELETE` and `TRUNCATE`**

   * `DELETE` → removes rows **with WHERE clause**, slower, logs changes
   * `TRUNCATE` → removes all rows, faster, cannot rollback in some DBs

10. **Difference between `DROP` and `ALTER`**

    * `DROP` → deletes table/index/database
    * `ALTER` → modifies table structure

---

### **Basic Queries**

11. **Select all columns from a table**

    ```sql
    SELECT * FROM employees;
    ```

12. **Select specific columns**

    ```sql
    SELECT name, salary FROM employees;
    ```

13. **Filter rows using `WHERE`**

    ```sql
    SELECT * FROM employees WHERE salary > 50000;
    ```

14. **Order data**

    ```sql
    SELECT * FROM employees ORDER BY salary DESC;
    ```

15. **Remove duplicates**

    ```sql
    SELECT DISTINCT department FROM employees;
    ```

16. **Count rows**

    ```sql
    SELECT COUNT(*) FROM employees;
    ```

17. **Aggregate functions**

    ```sql
    SELECT AVG(salary), MAX(salary), MIN(salary) FROM employees;
    ```

18. **Group by a column**

    ```sql
    SELECT department, COUNT(*) FROM employees GROUP BY department;
    ```

19. **String matching using `LIKE`**

    ```sql
    SELECT * FROM employees WHERE name LIKE 'A%';
    ```

20. **Using `IN` and `NOT IN`**

    ```sql
    SELECT * FROM employees WHERE department IN ('HR','IT');
    ```

---

### **Joins**

21. **What is a JOIN in SQL?**

    * Combines rows from **two or more tables** based on a related column.

22. **Inner Join example**

    ```sql
    SELECT e.name, d.name
    FROM employees e
    INNER JOIN departments d ON e.department_id = d.id;
    ```

23. **Left Join example**

    * Returns **all rows from left table** and matching from right.

    ```sql
    SELECT e.name, d.name
    FROM employees e
    LEFT JOIN departments d ON e.department_id = d.id;
    ```

24. **Right Join example**

    * Returns all rows from right table and matching from left.

25. **Full Outer Join example**

    * Returns **all rows** from both tables, NULL where no match.

26. **Self Join**

    ```sql
    SELECT e1.name AS Employee, e2.name AS Manager
    FROM employees e1
    LEFT JOIN employees e2 ON e1.manager_id = e2.id;
    ```

27. **Cross Join**

    * Returns **cartesian product** of two tables.

28. **Natural Join**

    * Joins tables **automatically on columns with same names**.

29. **Difference between INNER JOIN and LEFT JOIN**

    * INNER → only matching rows
    * LEFT → all left rows + matching right

30. **Difference between JOIN and UNION**

    * JOIN → combines columns from tables
    * UNION → combines rows of **two queries with same structure**

---

### **Constraints & Indexes**

31. **What is a unique key?**

    * Ensures **all values in a column are unique**, can have NULLs (depending on DB).

32. **What is an index?**

    * Improves **query performance** by allowing faster lookups.

33. **Primary key vs Unique key**

    * Primary → unique + not null
    * Unique → unique, can be null

34. **What is a check constraint?**

    ```sql
    CREATE TABLE employees (
        salary INT CHECK (salary > 0)
    );
    ```

35. **What is a default value constraint?**

    * Assigns **default value** if none provided:

    ```sql
    CREATE TABLE employees (
        status VARCHAR(10) DEFAULT 'Active'
    );
    ```

36. **What is an auto-increment column?**

    * Automatically generates **sequential numeric values**.

37. **Difference between clustered and non-clustered index**

    * Clustered → sorts data physically
    * Non-clustered → separate structure, logical pointers

38. **What is a composite key?**

    * **Combination of two or more columns** as primary key.

39. **What is a surrogate key?**

    * Artificial unique key, usually auto-increment.

40. **What is referential integrity?**

    * Ensures **foreign key relationships** are valid (child cannot reference non-existing parent).

---

## Medium SQL Questions

### **Joins & Subqueries**

41. **What is a correlated subquery?**

* A subquery that **depends on the outer query** for its values.

```sql
SELECT e.name
FROM employees e
WHERE salary > (SELECT AVG(salary) 
                FROM employees 
                WHERE department_id = e.department_id);
```

42. **Difference between subquery and join**

* Subquery → query inside another query
* Join → combines tables in a single query

43. **What is a derived table?**

* A **subquery used as a table** in FROM clause.

```sql
SELECT * FROM (SELECT * FROM employees WHERE salary>50000) AS high_paid;
```

44. **Nested subqueries example**

* Subquery inside another subquery for complex filtering.

45. **EXISTS vs IN**

* `EXISTS` → returns TRUE if subquery returns rows
* `IN` → matches values in a list

46. **Difference between ANY and ALL**

* `ANY` → comparison with **any value in subquery**
* `ALL` → comparison with **all values in subquery**

47. **Using COALESCE**

* Returns **first non-NULL value**

```sql
SELECT COALESCE(NULL, NULL, 'Hello'); -- Hello
```

48. **Using CASE statement**

```sql
SELECT name,
       CASE WHEN salary>50000 THEN 'High'
            ELSE 'Low'
       END AS salary_level
FROM employees;
```

49. **What is a common table expression (CTE)?**

* Temporary result set referenced **within a query using WITH**

```sql
WITH HighSalary AS (
    SELECT * FROM employees WHERE salary>50000
)
SELECT * FROM HighSalary;
```

50. **Recursive CTE example**

* Useful for hierarchical data, e.g., employee-manager chain.

---

### **Views & Stored Procedures**

51. **What is a view?**

* Virtual table based on **SELECT query**, does not store data physically.

```sql
CREATE VIEW HighSalary AS SELECT * FROM employees WHERE salary>50000;
```

52. **Difference between view and table**

* Table → stores data
* View → stores query, data fetched dynamically

53. **What is a materialized view?**

* Stores **actual data**, updated periodically, faster than regular views.

54. **What is a stored procedure?**

* Predefined SQL code stored in DB, **can accept parameters and perform operations**.

```sql
CREATE PROCEDURE GetEmployee(IN dept_id INT)
BEGIN
    SELECT * FROM employees WHERE department_id = dept_id;
END;
```

55. **What is a function in SQL?**

* Returns a single value, can be **user-defined or built-in**.

56. **Difference between function and stored procedure**

* Function → returns a value, can be used in SELECT
* Procedure → may not return value, used for operations

57. **What is a trigger?**

* Automatically executes **on INSERT, UPDATE, DELETE**.

```sql
CREATE TRIGGER before_insert_employee
BEFORE INSERT ON employees
FOR EACH ROW
SET NEW.created_at = NOW();
```

58. **Difference between BEFORE and AFTER trigger**

* BEFORE → executes **before operation**
* AFTER → executes **after operation**

59. **What is a cursor?**

* Pointer to **iterate over query results row by row**

60. **Dynamic SQL**

* SQL generated **at runtime** using variables or code.

---

### **Transactions & Locks**

61. **What is a transaction in SQL?**

* A sequence of operations treated as a **single unit**, which **commits** or **rolls back** entirely.

62. **ACID properties**

* **Atomicity:** all or nothing
* **Consistency:** database stays consistent
* **Isolation:** transactions don’t interfere
* **Durability:** committed data persists

63. **Commands to manage transactions**

```sql
BEGIN TRANSACTION;
COMMIT;
ROLLBACK;
```

64. **Isolation levels**

* Read Uncommitted, Read Committed, Repeatable Read, Serializable

65. **Deadlock in SQL**

* Two transactions **wait for each other** indefinitely; DB resolves using lock mechanisms.

66. **Lock types**

* Shared (read), Exclusive (write)

67. **Difference between implicit and explicit transactions**

* Implicit → auto-commit after each statement
* Explicit → controlled using BEGIN, COMMIT, ROLLBACK

68. **SAVEPOINT example**

* Create points in transaction to **rollback partially**:

```sql
SAVEPOINT sp1;
ROLLBACK TO sp1;
```

69. **Difference between COMMIT and ROLLBACK**

* COMMIT → saves changes permanently
* ROLLBACK → undoes changes

70. **What is autocommit?**

* Each statement **committed automatically** unless wrapped in explicit transaction.

---

### **Intermediate SQL Functions & Queries**

71. **Aggregate functions**

* `SUM()`, `AVG()`, `COUNT()`, `MAX()`, `MIN()`

72. **Window functions**

* Operate over **a partition of rows**, e.g.,

```sql
SELECT name, salary,
       RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
FROM employees;
```

73. **ROW_NUMBER vs RANK**

* ROW_NUMBER → unique sequential number
* RANK → same rank for ties, may skip numbers

74. **What is GROUPING SETS**

* Generate multiple groupings in one query

75. **Pivot and Unpivot**

* Pivot → convert rows → columns
* Unpivot → convert columns → rows

---

## Hard SQL Questions

### **Advanced Queries & Joins**

76. **What is a recursive query?**

* A query that **references itself**, often used to handle hierarchical data.

```sql
WITH RECURSIVE EmployeeHierarchy AS (
    SELECT id, manager_id, name FROM employees WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.manager_id, e.name
    FROM employees e
    INNER JOIN EmployeeHierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM EmployeeHierarchy;
```

77. **Difference between correlated and non-correlated subquery**

* Correlated → depends on outer query
* Non-correlated → independent, executes once

78. **Using advanced joins (self-join, cross-join, natural join)**

* Self-join → joining a table with itself
* Cross-join → cartesian product
* Natural join → joins automatically on same column names

79. **Anti-join**

* Returns **rows in one table that do not have a match** in another:

```sql
SELECT * FROM employees e
WHERE NOT EXISTS (SELECT 1 FROM departments d WHERE e.department_id = d.id);
```

80. **Semi-join**

* Returns rows in one table **if a match exists** in another (using `EXISTS`).

---

### **Indexing & Performance**

81. **What is an index and types**

* Improves query performance by allowing **faster lookups**.
* Types: Clustered, Non-clustered, Unique, Composite, Full-text

82. **Clustered vs non-clustered index**

* Clustered → data stored physically in index order
* Non-clustered → index separate from data

83. **What is a covering index?**

* Contains **all columns** required by a query → avoids accessing the table

84. **When to use an index**

* Frequently queried columns, primary/foreign keys, columns in WHERE, ORDER BY, JOIN

85. **Drawbacks of indexing**

* Slower inserts/updates/deletes, storage overhead

86. **How to optimize slow queries**

* Use indexes, avoid `SELECT *`, optimize joins, avoid correlated subqueries

87. **Explain execution plan**

* Shows how DB executes query, helps identify bottlenecks

88. **What is query optimization**

* Modifying queries or schema to **improve execution performance**

89. **Partitioning in SQL**

* Dividing large tables into **smaller, manageable pieces**
* Types: Range, List, Hash, Composite

90. **Denormalization**

* Reducing joins by **adding redundant data** for faster reads

---

### **Advanced Functions & Analytical Queries**

91. **Window functions for ranking**

* `RANK()`, `DENSE_RANK()`, `ROW_NUMBER()`
* Operate over **partitions without collapsing rows**

92. **LEAD and LAG functions**

```sql
SELECT name, salary,
       LAG(salary,1) OVER (ORDER BY salary) AS prev_salary,
       LEAD(salary,1) OVER (ORDER BY salary) AS next_salary
FROM employees;
```

93. **CUME_DIST and PERCENT_RANK**

* Calculate **percentile ranking** in a dataset

94. **NTH_VALUE and FIRST_VALUE**

* Return **nth value or first value** in a partition

95. **Advanced GROUP BY**

* `ROLLUP`, `CUBE`, `GROUPING SETS` to get multiple aggregations

96. **PIVOT example**

* Convert rows into columns:

```sql
SELECT *
FROM (SELECT department, month, revenue FROM sales) src
PIVOT (SUM(revenue) FOR month IN ([Jan],[Feb],[Mar])) pvt;
```

97. **UNPIVOT example**

* Convert columns into rows

98. **Handling recursive hierarchical data**

* Using **self-joins or recursive CTEs** for org charts

---

### **Transactions, Locks & Concurrency**

99. **Deadlock handling**

* Detect using DB tools, **rollback one transaction**, retry later

100. **Optimistic vs pessimistic locking**

* Optimistic → assumes no conflict, checks before commit
* Pessimistic → locks resources to prevent conflict