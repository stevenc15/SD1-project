import DashboardBox from '@/components/DashboardBox'
import { useEffect, useRef } from 'react'
import Highcharts from 'highcharts';
import { useTheme } from '@mui/material'

type Props = {
  emg_data: any;
  imu_data: any
}

const Row4 = ({emg_data, imu_data}: Props) => {
  const emgChartRef = useRef<HTMLDivElement>(null)
  const imuAccChartRef = useRef<HTMLDivElement>(null)
  const imuGyroChartRef = useRef<HTMLDivElement>(null)
  const palette = useTheme()
  useEffect( () => {
    if (emg_data && emgChartRef.current) {
      const data = emg_data['time'].map((time: any, index: number) => [time, emg_data['IM EMG4'][index]])
      const options: Highcharts.Options = {
        chart: {
            zooming: {
                type: 'x'
            },
            backgroundColor: palette.palette.grey[900],
        },
    
        title: {
            text: 'Sensor 4 EMG',
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
            tickColor: "#ffff"
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
          min: -0.0006,
          max: 0.0006
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
            data: data,
            lineWidth: 0.5,
            name: 'EMG SIGNALS',
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
            text: 'S4 ACC',
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
          min: -20000,
          max: 5000
        },
        legend: {
          itemStyle: {
            color: "#ffff"
          }
        },
        credits: {
          enabled: false
        },
        series: [
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCX4'][index]]),
            lineWidth: 1,
            name: 'ACCX4',
            color: palette.palette.secondary[600]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCY4'][index]]),
            lineWidth: 1,
            name: 'ACCY4',
            color: palette.palette.grey[100]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCZ4'][index]]),
            lineWidth: 1,
            name: 'ACCZ4',
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
            text: 'S4 GYRO',
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
        },
        legend: {
          itemStyle: {
            color: "#ffff"
          }
        },
        credits: {
          enabled: false
        },
        series: [
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['GYROX4'][index]]),
            lineWidth: 1,
            name: 'GYROX4',
            color: palette.palette.secondary[600]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['GYROY4'][index]]),
            lineWidth: 1,
            name: 'GYROY4',
            color: palette.palette.grey[100]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['GYROZ4'][index]]),
            lineWidth: 1,
            name: 'GYROZ4',
            color: palette.palette.sky[500]
          }
        ]
    };
    Highcharts.chart(imuGyroChartRef.current, options)
    }
  }, [emg_data, palette, imu_data])
  return (
    <>
      <DashboardBox gridArea="m" ref={emgChartRef}></DashboardBox>
      <DashboardBox gridArea="n" ref={imuAccChartRef}></DashboardBox>
      <DashboardBox gridArea="o" ref={imuGyroChartRef}></DashboardBox>
    </>
  )
}

export default Row4