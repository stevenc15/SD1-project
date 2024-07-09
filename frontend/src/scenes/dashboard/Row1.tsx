import DashboardBox from '@/components/DashboardBox'
import LiveChart from '@/components/LiveChart'
import React from 'react'

type Props = {}

const Row1 = (props: Props) => {
  return (
    <>
        <DashboardBox  gridArea="a"> <LiveChart /></DashboardBox>
        <DashboardBox  gridArea="b">More data goes here</DashboardBox>
        <DashboardBox  gridArea="c"></DashboardBox>
    </>
  )
}

export default Row1