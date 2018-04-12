# Nth-to-Default-CDS-Basket

The project was originally developed in Microsoft Excel.

Although the Excel implementation correctly priced the product, the file was overly slow for high number of simulations.

That is the main reason for implementing the project in Python.

This implementation has one simplification, i.e. the premium leg would  require the single possible cashflows to be individually 
present-valued using the appropriate discount factors (depending on the different time the cashflow is produced in the simulation).  This
aspect will be improved in future implementations.

The project is based on a simulation that uses a Gaussian Copula. Further information on this are available

