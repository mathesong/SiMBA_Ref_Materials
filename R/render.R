library(here)
library(rmarkdown)

setwd(here::here("R"))

render("SiMBA_SRTM_Demonstration.Rmd", output_format = "all")

file.rename("SiMBA_SRTM_Demonstration.md", "README.md")

setwd(here::here())
