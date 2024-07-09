// src/components/LiveChart.tsx
import React, { useRef, useEffect } from 'react';
import Highcharts, { ChartOptions } from 'highcharts';

let chart: Highcharts.Chart | undefined;

const LiveChart: React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);

  async function requestData() {
    const result = await fetch('http://127.0.0.1:5000/data');
    if (result.ok) {
      const data = await result.json();
      console.log('Fetched data:', data);

      const [date, value] = data;
      const point = [new Date(date).getTime(), value * 10];
      const series = chart?.series[0];
      if (series) {
        const shift = series.data.length > 20; // shift if the series is longer than 20
        // add the point
        series.addPoint(point, true, shift);
      }
      // call it again after one second
      setTimeout(requestData, 1000);
    }
  }

  useEffect(() => {
    if (containerRef.current) {
      const options: Highcharts.Options = {
        chart: {
          type: 'spline',
          events: {
            load: requestData
          }
        },
        title: {
          text: 'Live random data'
        },
        xAxis: {
          type: 'datetime',
          tickPixelInterval: 150,
          // maxZoom: 20 * 1000
          minRange: 20 * 1000
        },
        yAxis: {
          minPadding: 0.2,
          maxPadding: 0.2,
          title: {
            text: 'Value',
            margin: 80
          }
        },
        series: [{
          type: 'spline',
          name: 'Random data',
          data: []
        }]
      };

      chart = Highcharts.chart(containerRef.current, options);
    }

    return () => {
      if (chart) {
        chart.destroy();
      }
    };
  }, []);

  return (
    <div ref={containerRef} id="container"></div>
  );
};

export default LiveChart;
