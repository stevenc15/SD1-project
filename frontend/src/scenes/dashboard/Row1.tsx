import DashboardBox from '@/components/DashboardBox'
import LiveChart from '@/components/LiveChart'
import StaticGraph from '@/components/StaticGraph'
import React from 'react'
import SampleData from '@/components/SampleData'

type Props = {}

const Row1 = (props: Props) => {
  return (
    <>
        <DashboardBox gridArea="a"></DashboardBox>
        <DashboardBox gridArea="b">{/*<LiveChart /> <StaticGraph />*/}</DashboardBox>
        <DashboardBox gridArea="c"> {/*<SampleData></SampleData>*/}</DashboardBox>
    </>
  )
}

export default Row1