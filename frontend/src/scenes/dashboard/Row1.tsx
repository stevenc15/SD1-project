import DashboardBox from '@/components/DashboardBox'
import LiveChart from '@/components/LiveChart'
import StaticGraph from '@/components/StaticGraph'
import React, { useEffect, useRef } from 'react'
import SampleData from '@/components/SampleData'
import Highcharts from 'highcharts';
import { useTheme } from '@mui/material'

type Props = {
  emg_data: any;
  imu_data: any;
}

const Row1 = ({emg_data, imu_data}: Props) => {
  const emgChartRef = useRef<HTMLDivElement>(null)
  const imuAccChartRef = useRef<HTMLDivElement>(null)
  const imuGyroChartRef = useRef<HTMLDivElement>(null)
  const palette = useTheme()

  useEffect( () => {
    if (emg_data && emgChartRef.current) {
      const data = emg_data['time'].map((time: any, index: number) => [time, emg_data['IM EMG1'][index]])
      const options: Highcharts.Options = {
        chart: {
            zooming: {
                type: 'x'
            },
            backgroundColor: palette.palette.grey[900],
        },
    
        title: {
            text: 'Sensor 1 EMG',
            align: 'left',
            style: {
              color: "#ffff"
            }
        },
    
        accessibility: {
            screenReaderSection: {
                beforeChartFormat: '<{headingTagName}>' +
                    '{chartTitle}</{headingTagName}><div>{chartSubtitle}</div>' +
                    '<div>{chartLongdesc}</div><div>{xAxisDescription}</div><div>' +
                    '{yAxisDescription}</div>'
            }
        },
    
        // tooltip: {
        //     valueDecimals: 2
        // },
    
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

        yAxis: {
          title: {
            text: "mV",
            style: {
              color: "#ffff"
            }
          },
          labels: {
            style: {
              color: "#ffff"
            }
          },
          min: -0.0005,
          max: 0.00045
        },
        legend: {
          itemStyle: {
            color: "#ffff"
          }
        },
        series: [{
            type: 'spline',
            data: data,
            lineWidth: 0.5,
            name: 'EMG1 SIGNALS',
        }]
    };
    Highcharts.chart(emgChartRef.current, options)
    }

    if (imu_data && imuAccChartRef.current) {
      // const data = imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCX1'][index]])
      const options: Highcharts.Options = {
        chart: {
            zooming: {
                type: 'x'
            },
            backgroundColor: palette.palette.grey[900],
        },
    
        title: {
            text: 'S1 ACC',
            align: 'left',
            style: {
              color: "#ffff"
            }
        },
    
        accessibility: {
            screenReaderSection: {
                beforeChartFormat: '<{headingTagName}>' +
                    '{chartTitle}</{headingTagName}><div>{chartSubtitle}</div>' +
                    '<div>{chartLongdesc}</div><div>{xAxisDescription}</div><div>' +
                    '{yAxisDescription}</div>'
            }
        },
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

        yAxis: {
          title: {
            text: "mV",
            style: {
              color: "#ffff"
            }
          },
          labels: {
            style: {
              color: "#ffff"
            }
          },
          min: -28000,
          max: 20000
        },
        legend: {
          itemStyle: {
            color: "#ffff"
          }
        },
        series: [
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCX1'][index]]),
            lineWidth: 1,
            name: 'ACCX1',
            color: palette.palette.secondary[600]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCY1'][index]]),
            lineWidth: 1,
            name: 'ACCY1',
            color: palette.palette.grey[100]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCZ1'][index]]),
            lineWidth: 1,
            name: 'ACCZ1',
            color: palette.palette.sky[500]
          }
        ]
    };
    Highcharts.chart(imuAccChartRef.current, options)
    }
    if (imu_data && imuGyroChartRef) {
      const options: Highcharts.Options = {
        chart: {
            zooming: {
                type: 'x'
            },
            backgroundColor: palette.palette.grey[900],
        },
    
        title: {
            text: 'S1 GYRO',
            align: 'left',
            style: {
              color: "#ffff"
            }
        },
    
        accessibility: {
            screenReaderSection: {
                beforeChartFormat: '<{headingTagName}>' +
                    '{chartTitle}</{headingTagName}><div>{chartSubtitle}</div>' +
                    '<div>{chartLongdesc}</div><div>{xAxisDescription}</div><div>' +
                    '{yAxisDescription}</div>'
            }
        },
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

        yAxis: {
          title: {
            text: "mV",
            style: {
              color: "#ffff"
            }
          },
          labels: {
            style: {
              color: "#ffff"
            }
          },
          min: -500,
          max: 500
        },
        legend: {
          itemStyle: {
            color: "#ffff"
          }
        },
        series: [
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['GYROX1'][index]]),
            lineWidth: 1,
            name: 'GYROX1',
            color: palette.palette.secondary[600]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['GYROY1'][index]]),
            lineWidth: 1,
            name: 'GYROY1',
            color: palette.palette.grey[100]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['GYROZ1'][index]]),
            lineWidth: 1,
            name: 'GYROZ1',
            color: palette.palette.sky[500]
          }
        ]
    };
    Highcharts.chart(imuGyroChartRef.current, options)
    }

  }, [emg_data, palette, imu_data])
  return (
    <>
        <DashboardBox gridArea="a" ref={emgChartRef}></DashboardBox>
        <DashboardBox gridArea="b" ref={imuAccChartRef}></DashboardBox>
        <DashboardBox gridArea="c" ref={imuGyroChartRef}> </DashboardBox>
    </>
  )
}

export default Row1