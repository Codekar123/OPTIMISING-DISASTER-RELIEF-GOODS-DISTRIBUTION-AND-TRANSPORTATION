# OPTIMISING-DISASTER-RELIEF-GOODS-DISTRIBUTION-AND-TRANSPORTATION
Built a Python tool to plan faster disaster relief delivery routes using Linear Programming and Ant Colony Optimization. Aimed at making emergency logistics smarter and more efficient.
# Disaster Relief Routing Optimization

This project models and solves the problem of optimally routing disaster relief goods from warehouses to affected regions using:

- **Linear Programming (LP)** for small-to-moderate problem sizes (using PuLP in Python)
- **Ant Colony Optimization (ACO)** for larger, more complex scenarios

### 🔧 Tools & Technologies
- Python
- PuLP (for LP modeling)
- NumPy
- Random module (for ACO simulation)

### 📂 Contents
- `LP_DR_OPTI.py`: Linear programming solution using PuLP
- `ForLargeNoOfVaraibles_usingACO.py`: Metaheuristic ACO solution
- `DAY2_P4.pdf`: Full project report detailing models, constraints, and implementation

### 👥 Contributors
- Sai Ganesh 
- Sathwik Thupakula 

---

### 🚀 How to Run
```bash
# For LP model
python LP_DR_OPTI.py

# For ACO model
python ForLargeNoOfVaraibles_usingACO.py

