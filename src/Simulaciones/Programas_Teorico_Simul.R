##Sumulate AR(1) with intercept ----

simulate_ar1 <- function(n, intercept, ar_coefficient, sd_error, burn_in = 0) {
  # Initialize the vector to hold the AR(1) data
  ar_data <- numeric(n + burn_in)  # Increase size to accommodate burn-in
  
  # Set the first value (can be any value, here we set it to the intercept)
  ar_data[1] <- intercept + rnorm(1, mean = 0, sd = sd_error)
  
  # Simulate the AR(1) process
  for (t in 2:(n + burn_in)) {
    ar_data[t] <- intercept + ar_coefficient * ar_data[t - 1] + rnorm(1, mean = 0, sd = sd_error)
  }
  
  # Return only the data after the burn-in period
  return(ar_data[(burn_in + 1):(n + burn_in)])
}

# Example usage
set.seed(123) # Setting seed for reproducibility
simulated_data <- simulate_ar1(n = 100, intercept = 5, ar_coefficient = 0.8, sd_error = 1, burn_in = 10)
plot(simulated_data, type = 'l', main = 'Simulated AR(1) Process with Burn-in', ylab = 'Value', xlab = 'Time')

###Simulate MA(1) with Intercept ----
simulate_ma1 <- function(n, mean = 0, ma_coefficient, sd_error, burn_in = 0) {
  # Initialize the vector to hold the MA(1) data
  ma_data <- numeric(n + burn_in)  # Increase size to accommodate burn-in
  error_terms <- rnorm(n + burn_in, mean = 0, sd = sd_error)  # Store error terms
  
  # Simulate the MA(1) process
  for (t in 1:(n + burn_in)) {
    if (t == 1) {
      ma_data[t] <- mean + error_terms[t]  # First observation is mean + error
    } else {
      ma_data[t] <- mean + error_terms[t] + ma_coefficient * error_terms[t - 1]  # MA(1) formula
    }
  }
  
  # Return only the data after the burn-in period
  return(ma_data[(burn_in + 1):(n + burn_in)])
}


set.seed(123)  # Setting seed for reproducibility
simulated_data_ma1 <- simulate_ma1(n = 200, mean = 5, ma_coefficient = 0.5, sd_error = 1, burn_in = 10)
plot(simulated_data_ma1, type = 'l', main = 'Simulated MA(1) Process with Burn-in', ylab = 'Value', xlab = 'Time')


####Get first maxLag of the theoretical autocovariance function for arma models ----
##### AR(1) with phi=0.5 and sigma^2=1 ----
ltsa::tacvfARMA(phi = c(0.5), maxLag = 10, sigma2 = 1)


##### MA(1) with theta=-0.5 and sigma^2=1 ----
ltsa::tacvfARMA(theta = c(-0.5), maxLag = 10, sigma2 = 1)
