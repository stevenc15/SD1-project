import DashboardBox from '@/components/DashboardBox'
import { useEffect, useRef } from 'react'
import Highcharts from 'highcharts';
import { useTheme } from '@mui/material'


let chart: Highcharts.Chart;
const LiveRow = () => {
  const palette = useTheme()
  const containerRef = useRef<HTMLDivElement>(null);

  async function requestData() {

    const result = await fetch('http://127.0.0.1:5000/pred_data');
    if (result.ok) {
      const data = await result.json();

      const {time, elbow_flex_r_pred, elbow_flex_r} = data;
      // console.log("This is the date: " + date)
      // console.log("This is the value: " + value)
      // const point = [new Date(date).getTime()];
      const point_truth = [time, elbow_flex_r];
      const point_pred = [time, elbow_flex_r_pred];

      if (chart) {
        const series_truth = chart.series[0];
        const series_pred = chart.series[1];
        const shift = series_truth.data.length > 300; // shift if the series is longer than 20

        // add the point
        series_truth.addPoint(point_truth, true, shift);
        series_pred.addPoint(point_pred, true, shift);
        
      }
      // call it again after one second (1000)
      // setTimeout(requestData, 1000);
      setTimeout(requestData, 100);
    }
    else if (result.status === 400) {
      initializeChart();
    }
  }

  function initializeChart() {
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
                [0, palette.palette.grey[900]],
                [1, palette.palette.grey[900]]
            ]
          },
          allowMutatingData: false,
        },
        title: {
          text: 'Ground Truth vs Predicted Value',
          align: 'center',
          style: {
            color: "#ffff"
          }
      },
      // Bigger size options for the truth vs predictions plot
      xAxis: {
        title: {
          text: "Time (s)",
          style: {
            color: "#ffff"
          }
        },
          type: 'linear',
          labels: {
            style: {
              color: "#ffff"
            }
          },
          lineColor: "#ffff",
          tickColor: "#ffff",
          // tickInterval: 0.01, // this property modifies the data's domain
          // categories: emg_data['time']
      },
    
        // xAxis: {
        //   type: 'linear',
        //   title: {
        //     text: 'Time (s)'
        //   },
        //   tickPixelInterval: 150,
        //   minRange: 20,
        //   labels: {
        //     format: '{value} s'
        //   }
        // },

        yAxis: {
          minPadding: 0.2,
          maxPadding: 0.2,
          title: {
            text: 'Degrees',
            margin: 80,
            style: {
              color: "#ffff"
            }
          },
          labels: {
            style: {
              color: "#ffff"
            }
          },
          gridLineWidth: 0 // horizontal white lines within the chart
        },
        legend: {
          itemStyle: {
            color: "#ffff"
          }
        },
        credits: {
          enabled: false
        },
        series: [{
          type: 'spline',
          name: 'Elbow Truth',
          data: []
        },
        {
          type: 'spline',
          name: 'Elbow Prediction',
          data: []
        }]
      };

      chart = Highcharts.chart(containerRef.current, options);
    }
  };
  
  useEffect( () => {
    initializeChart();
    return () => {
      if (chart) {
        chart.destroy();
      }
    };
  },)

  return (
    <>
      <DashboardBox ref={containerRef} height={600} width={670}></DashboardBox>
    </>
  )
}

export default LiveRow