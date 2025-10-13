
CREATE TABLE time (
    time_id SERIAL PRIMARY KEY,
    date VARCHAR(50) NOT NULL,
    hour INT NOT NULL,
    day INT NOT NULL,
    day_of_week VARCHAR(20),
    month INT NOT NULL,
    year INT NOT NULL
);

CREATE TABLE station (
    station_id  SERIAL PRIMARY KEY,
    station_name VARCHAR(100) NOT NULL,
    latitude DECIMAL(9,6) NOT NULL,
    longitude DECIMAL(9,6) NOT NULL
);

CREATE TABLE users (
    user_id  SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
	age INT,
    gender VARCHAR(10),
    user_type VARCHAR(50)
);

CREATE TABLE trips (
    trip_id  SERIAL PRIMARY KEY,
    start_time VARCHAR(50) NOT NULL,
    end_time VARCHAR(50) NOT NULL,
    duration_sec INT NOT NULL,
    bike_id INT NOT NULL,
    start_station_id INT NOT NULL,
    end_station_id INT NOT NULL,
    user_id INT NOT NULL,
    time_id INT NOT NULL,

    CONSTRAINT fk_start_station FOREIGN KEY (start_station_id)
        REFERENCES station(station_id),

    CONSTRAINT fk_end_station FOREIGN KEY (end_station_id)
        REFERENCES station(station_id),

    CONSTRAINT fk_user FOREIGN KEY (user_id)
        REFERENCES users(user_id),

    CONSTRAINT fk_time FOREIGN KEY (time_id)
        REFERENCES time(time_id)
);