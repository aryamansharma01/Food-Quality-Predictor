#include <stdlib.h>
#include <stdint.h>

#include <Arduino.h>


// #define cli() asm volatile("cli"::)
// #define sei() asm volatile("sei"::)


typedef struct
{
	uint64_t start;
	uint64_t stop;
}
Timer;


static Timer timer = { 0 };


void timer_init()
{
}


void timer_start()
{
	timer.start = micros();
}


void timer_stop()
{
	timer.stop = micros();
}

uint64_t timer_value_us()
{
	return timer.stop - timer.start;
}
