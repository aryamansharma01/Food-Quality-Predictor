#include <string.h>

#include "user_app.h"
#include "timer.h"

#include "neuton/calculator.h"

typedef struct
{
	uint32_t        reverseByteOrder;
	CalculatorStats stats;
	float			times[10];
	uint32_t		bufferLength;
	float			sum;
	uint32_t		pos;
}
AppContext;


static AppContext app = { 0 };
static NeuralNet neuralNet = { 0 };
static uint32_t memUsage = 0;
static float usSample = 0;

extern const unsigned char model_bin[];
extern const unsigned int model_bin_len;

void led_green(uint8_t state);
void led_red(uint8_t state);


inline ModelInfo app_model_info()
{
	ModelInfo info;

	info.rowsCount = neuralNet.outputsDim;
	info.taskType = neuralNet.taskType;

	return info;
}


inline int app_dataset_info(DatasetInfo* info)
{
	app.reverseByteOrder = info->reverseByteOrder;
	return info->rowsCount == app_inputs_size() ? 0 : 1;
}


inline uint32_t app_inputs_size()
{
	return neuralNet.inputsDim;
}


inline uint32_t app_model_size()
{
	return model_bin_len;
}


inline void app_reset_stats(CalculatorStats *stats)
{
	stats->usSampleMin = 1000000000.0;
	stats->usSampleMax = 0.0;
	stats->usSampleAvg = 0.0;
}


inline CalculatorStats app_get_stats()
{
	return app.stats;
}


inline void app_reset()
{
	app_reset_stats(&app.stats);

	memset(app.times, 0, sizeof(app.times));
	app.sum = 0;
	app.pos = 0;
	app.bufferLength = sizeof(app.times) / sizeof(app.times[0]);

	led_green(0);
	led_red(0);
}

#if defined(NEUTON_MEMORY_BENCHMARK)
uint32_t _NeutonExtraMemoryUsage()
{
	return memUsage;
}
#endif

uint8_t app_init()
{
	return (ERR_NO_ERROR != CalculatorInit(&neuralNet, NULL));
}


static inline void Reverse4BytesValuesBuffer(void* buf, uint32_t valuesCount)
{
	uint32_t* n = buf;
	uint32_t i;

	for (i = 0; i < valuesCount; ++i)
	{
		*n = (*n & 0x000000FFu) << 24 | (*n & 0x0000FF00u) << 8 | (*n & 0x00FF0000u) >> 8 | (*n & 0xFF000000u) >> 24;
		++n;
	}
}


inline float* app_run_inference(float* sample, uint32_t size_in, uint32_t *size_out)
{
	if (!sample || !size_out)
		return NULL;

	if (size_in / sizeof(float) != app_inputs_size())
		return NULL;

	*size_out = sizeof(float) * neuralNet.outputsDim;

	if (app.reverseByteOrder)
		Reverse4BytesValuesBuffer(sample, app_inputs_size());

	return CalculatorRunInference(&neuralNet, sample);
}

inline Err CalculatorOnInit(NeuralNet* neuralNet)
{
	memUsage += sizeof(*neuralNet);

	app_reset();
	timer_init();

	return CalculatorLoadFromMemory(neuralNet, model_bin, model_bin_len, 0);
}

inline void CalculatorOnFree(NeuralNet* neuralNet)
{

}

inline Err CalculatorOnLoad(NeuralNet* neuralNet)
{
	return ERR_NO_ERROR;
}


inline Err CalculatorOnRun(NeuralNet* neuralNet)
{
	return ERR_NO_ERROR;
}


inline void CalculatorOnInferenceStart(NeuralNet* neuralNet)
{
	timer_start();
}


inline void CalculatorOnInferenceEnd(NeuralNet* neuralNet)
{
	timer_stop();

	usSample = timer_value_us();
	CalculatorStats *stats = &app.stats;

	if (stats->usSampleMin > usSample)
		stats->usSampleMin = usSample;

	if (stats->usSampleMax < usSample)
		stats->usSampleMax = usSample;

	if (app.sum == 0)
	{
		app.sum = usSample * (float) app.bufferLength;
		uint32_t i;
		for (i = 0; i < app.bufferLength; i++)
			app.times[i] = usSample;
	}

	app.sum = app.sum - app.times[app.pos] + usSample;
	app.times[app.pos++] = usSample;
	app.pos %= app.bufferLength;
	stats->usSampleAvg = app.sum / (float) app.bufferLength;
}


inline void CalculatorOnInferenceResult(NeuralNet* neuralNet, float* result)
{
	if (neuralNet->taskType == TASK_BINARY_CLASSIFICATION && neuralNet->outputsDim >= 2)
	{
		float* value = result[0] >= result[1] ? &result[0] : &result[1];
		if (*value > 0.5)
		{
			if (value == &result[0])
			{
				led_green(1);
				led_red(0);
			}
			else
			{
				led_green(0);
				led_red(1);
			}
		}
		else
		{
			led_green(0);
			led_red(0);
		}
	}
}
