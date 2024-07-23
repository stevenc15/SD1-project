import React, { useEffect, useRef } from 'react';
import Highcharts from 'highcharts';
import DashboardBox from './DashboardBox';

const SampleData: React.FC = () => {
  const chartRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    async function get_big_data(){
      try {
        // const result = await fetch('http://127.0.0.1:5000/large_data')
        const result = await fetch('http://127.0.0.1:5000/sensor1_emg')
        if (result.ok) {
          const data = await result.json()
          console.log("HERE IS THE DATA", data)
          if (chartRef.current) {
            const options: Highcharts.Options = {
              chart: {
                  zooming: {
                      type: 'x'
                  }
              },
          
              title: {
                  text: 'Sensor 1 EMG ' + data.length + ' points',
                  align: 'left'
              },
          
              accessibility: {
                  screenReaderSection: {
                      beforeChartFormat: '<{headingTagName}>' +
                          '{chartTitle}</{headingTagName}><div>{chartSubtitle}</div>' +
                          '<div>{chartLongdesc}</div><div>{xAxisDescription}</div><div>' +
                          '{yAxisDescription}</div>'
                  }
              },
          
              tooltip: {
                  valueDecimals: 2
              },
          
              xAxis: {
                  type: 'linear'
              },

              yAxis: {
                title: {
                  text: "mV"
                }
              },

              series: [{
                  type: 'spline',
                  data: data,
                  lineWidth: 0.5,
                  name: 'EMG SIGNALS'
              }]
          };
          console.timeEnd('line');
          Highcharts.chart(chartRef.current, options)
          }
        }
        else {
          console.error('Failed to fetch data:', result.statusText)
        }
      } catch (error) {
        console.error('Error Fetching data: ', error)
      }
    }
    get_big_data()
    console.time('line');
    
  }, [])
  return <DashboardBox ref={chartRef}></DashboardBox>;
}

export default SampleData