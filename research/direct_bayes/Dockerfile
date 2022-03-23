# Usage:
# docker build -f Dockerfile -t r - < Dockerfile
# docker run -it --rm -v $(pwd):/volume: -u docker  r bash
FROM r-base
RUN R -e "install.packages('statmod',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('invgamma',dependencies=TRUE, repos='http://cran.rstudio.com/')"
