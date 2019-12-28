#pragma once
#include <chrono>

using namespace std::chrono;

class MyTimer {
private:
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point stop;

public:
  void StartTimer();
  void StopTimer();
  long long GetDurationInSeconds();
  double GetDurationInSecondsAccurate();
  long long GetDurationInMilliseconds();
  long long GetDurationInMicroseconds();
};

void MyTimer::StartTimer()
{
  start = high_resolution_clock::now();
}

void MyTimer::StopTimer()
{
  stop = high_resolution_clock::now();
}

long long MyTimer::GetDurationInSeconds()
{
  auto duration = duration_cast<seconds>(stop - start);
  return duration.count();
}

double MyTimer::GetDurationInSecondsAccurate()
{
  auto duration = duration_cast<microseconds>(stop - start);
  return (double)duration.count() / 1000000;
}

long long MyTimer::GetDurationInMilliseconds()
{
  auto duration = duration_cast<milliseconds>(stop - start);
  return duration.count();
}

long long MyTimer::GetDurationInMicroseconds()
{
  auto duration = duration_cast<microseconds>(stop - start);
  return duration.count();
}