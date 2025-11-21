COPY cast_info FROM 'benchmark/cast_info.csv' (FORMAT 'csv', quote '"', delimiter ',', escape '"', header 1);
COPY info_type FROM 'benchmark/info_type.csv' (FORMAT 'csv', quote '"', delimiter ',', escape '"', header 1);
COPY movie FROM 'benchmark/movie.csv' (FORMAT 'csv', quote '"', delimiter ',', escape '"', header 1);
COPY movie_info FROM 'benchmark/movie_info.csv' (FORMAT 'csv', quote '"', delimiter ',', escape '"', header 1);
COPY "name" FROM 'benchmark/name.csv' (FORMAT 'csv', quote '"', delimiter ',', escape '"', header 1);
COPY review FROM 'benchmark/reviews.csv' (FORMAT 'csv', quote '"', delimiter ',', escape '"', header 1);
COPY role_type FROM 'benchmark/role_type.csv' (FORMAT 'csv', quote '"', delimiter ',', escape '"', header 1);
