import DashboardBox from '@/components/DashboardBox'
import React from 'react'

type Props = {}

const Row6 = (props: Props) => {
  return (
    <>
        <DashboardBox gridArea="u"> </DashboardBox>
        <DashboardBox gridArea="v"> </DashboardBox>
        <DashboardBox gridArea="w"> </DashboardBox>
    </>
  )
}

export default Row6