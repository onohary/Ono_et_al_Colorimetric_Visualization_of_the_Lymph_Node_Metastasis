#==== Read Data ================================================================
Pdot_total_data <- read.csv("data/Accumulated_Pdot_amount.csv")

#==== Hotteling's T2 test ======================================================
# Install the package
install.packages("Hotelling")

# Load the package
library(Hotelling)

#Data Prepossessing
Normal <- subset(Pdot_total_data, Day == "0", select = c("F8BT", "MEHPPV", "TPP_PFO"))
day12 <- subset(Pdot_total_data, Day == "12", select = c("F8BT", "MEHPPV", "TPP_PFO"))


# analysis
hotelling_Normal_day12_pval <- hotelling.test(x = Normal, y = day12)$pval

hotteling_results <- c(
  hotelling_Normal_day12_pval
)

# Create a character vector for column names
names_vector <- c("Normal_vs_day12")

# Assign names to the vector elements
names(hotteling_results) <- names_vector

# Print the final matrix
print(hotteling_results)


#==== t-test ===================================================================
t_test_F8BT_pval <- t.test(x = Normal$F8BT, y = Day12$F8BT)$p.value
t_test_MEHPPV_pval <- t.test(x = Normal$MEHPPV, y = Day12$MEHPPV)$p.value
t_test_TPP_PFO_pval <- t.test(x = Normal$TPP_PFO, y = Day12$TPP_PFO)$p.value

t_test_results <- c(
  t_test_F8BT_pval,
  t_test_MEHPPV_pval,
  t_test_TPP_PFO_pval
)


# Create a character vector for column names
names_vector <- c("F8BT", "MEHPPV", "TPP_PFO")

# Assign names to the vector elements
names(t_test_results) <- names_vector

# Print the final matrix
print(t_test_results)