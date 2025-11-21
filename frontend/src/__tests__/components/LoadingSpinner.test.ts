import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'

describe('LoadingSpinner', () => {
  it('renders properly', () => {
    const wrapper = mount(LoadingSpinner)
    expect(wrapper.exists()).toBe(true)
  })

  it('applies size prop correctly', () => {
    const wrapper = mount(LoadingSpinner, {
      props: { size: 'lg' },
    })
    expect(wrapper.classes()).toContain('lg')
  })

  it('applies color prop correctly', () => {
    const wrapper = mount(LoadingSpinner, {
      props: { color: 'primary' },
    })
    expect(wrapper.classes()).toContain('primary')
  })

  it('has default size when no prop provided', () => {
    const wrapper = mount(LoadingSpinner)
    expect(wrapper.classes()).toContain('md')
  })
})
