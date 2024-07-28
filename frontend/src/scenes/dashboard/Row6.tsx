import DashboardBox from '@/components/DashboardBox'
import { useEffect, useRef } from 'react'
import Highcharts from 'highcharts';
import { useTheme } from '@mui/material'

type Props = {
  emg_data: any;
  imu_data: any
}

const Row6 = ({emg_data, imu_data}: Props) => {
  const emgChartRef = useRef<HTMLDivElement>(null)
  const imuAccChartRef = useRef<HTMLDivElement>(null)
  const imuGyroChartRef = useRef<HTMLDivElement>(null)
  const palette = useTheme()
  useEffect( () => {
    if (emg_data && emgChartRef.current) {
      const data = emg_data['time'].map((time: any, index: number) => [time, emg_data['IM EMG6'][index]])
      const options: Highcharts.Options = {
        chart: {
            zooming: {
                type: 'x'
            },
            backgroundColor: palette.palette.grey[900],
        },
    
        title: {
            text: 'Sensor 6 EMG',
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
            text: "Volts",
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
          max: 0.0005
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
            text: 'S6 ACC',
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
            text: "mm/s^2",
            style: {
              color: "#ffff"
            }
          },
          labels: {
            style: {
              color: "#ffff"
            }
          },
          min: -15000,
          // max: 10000
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
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCX6'][index]]),
            lineWidth: 1,
            name: 'ACCX6',
            color: palette.palette.secondary[600]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCY6'][index]]),
            lineWidth: 1,
            name: 'ACCY6',
            color: palette.palette.grey[100]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['ACCZ6'][index]]),
            lineWidth: 1,
            name: 'ACCZ6',
            color: palette.palette.sky[500]
          }
        ]
    };
    Highcharts.chart(imuAccChartRef.current, options)
    }
    if (imu_data && imuGyroChartRef.current) {
      const options: Highcharts.Options = {
        chart: {
            zooming: {
                type: 'x'
            },
            backgroundColor: palette.palette.grey[900],
        },
    
        title: {
            text: 'S6 GYRO',
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
            text: "Degrees",
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
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['GYROX6'][index]]),
            lineWidth: 1,
            name: 'GYROX6',
            color: palette.palette.secondary[600]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['GYROY6'][index]]),
            lineWidth: 1,
            name: 'GYROY6',
            color: palette.palette.grey[100]
          },
          {
            type: 'spline',
            data: imu_data['time'].map((time: any, index: number) => [time, imu_data['GYROZ6'][index]]),
            lineWidth: 1,
            name: 'GYROZ6',
            color: palette.palette.sky[500]
          }
        ]
    };
    Highcharts.chart(imuGyroChartRef.current, options)
    }
  }, [emg_data, palette, imu_data])
  return (
    <>
      <DashboardBox gridArea="u" ref={emgChartRef}></DashboardBox>
      <DashboardBox gridArea="v" ref={imuAccChartRef}></DashboardBox>
      <DashboardBox gridArea="w" ref={imuGyroChartRef}></DashboardBox>
    </>
  )
}

export default Row6