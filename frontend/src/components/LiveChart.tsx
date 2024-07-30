// src/components/LiveChart.tsx
import React, { useRef, useEffect } from 'react';
import Highcharts from 'highcharts';
import DashboardBox from './DashboardBox';
import { useTheme } from '@mui/material';

let chart: Highcharts.Chart;

const LiveChart: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { palette } = useTheme();

  async function requestData() {
    const result = await fetch('http://127.0.0.1:5000/data');
    if (result.ok) {
      const data = await result.json();
      // console.log('Fetched data:', data);

      const [date, value] = data;
      // console.log("This is the date: " + date)
      // console.log("This is the value: " + value)
      // const point = [new Date(date).getTime()];
      const point = [new Date(date).getTime(), value * 10];
      const series = chart.series[0];
        const shift = series.data.length > 30; // shift if the series is longer than 20
        // add the point
        series?.addPoint(point, true, shift);
      // call it again after one second (1000)
      // setTimeout(requestData, 1000);
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
          },
          // backgroundColor: palette.grey[500],
          backgroundColor: {
            // linearGradient: [0, 0, 500, 500],
            linearGradient: { x1: 0, y1: 0, x2: 1, y2: 1 },
            stops: [
                [0, palette.grey[100]],
                [1, palette.grey[600]]
            ]
          },
          allowMutatingData: false,
        },
        title: {
          text: 'GYRO 1'
        },
        xAxis: {
          type: 'datetime',
          tickPixelInterval: 150,
          // maxZoom: 20 * 1000
          minRange: 20 * 1000,
        },
        yAxis: {
          minPadding: 0.2,
          maxPadding: 0.2,
          title: {
            text: 'Value',
            margin: 80
          },
          gridLineWidth: 0 // horizontal white lines within the chart
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
  });

  return <DashboardBox ref={containerRef} id="container"></DashboardBox>
};

export default LiveChart;
