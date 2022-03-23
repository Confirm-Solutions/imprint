#Gives the Taylor-based upper bound

taylorUpperBound = function(ucbf,ucbGradf,secondOrder,h){
   ucbf + ucbGradf + h^T %*% secondOrder %*% h
}
