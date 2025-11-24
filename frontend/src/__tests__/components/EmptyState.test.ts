import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import EmptyState from '@/components/common/EmptyState.vue'

describe('EmptyState', () => {
  it('renders title and description', () => {
    const wrapper = mount(EmptyState, {
      props: {
        title: 'No data',
        description: 'There is no data to display',
      },
    })

    expect(wrapper.text()).toContain('No data')
    expect(wrapper.text()).toContain('There is no data to display')
  })

  it('renders icon when provided', () => {
    const wrapper = mount(EmptyState, {
      props: {
        title: 'No data',
        icon: 'inbox',
      },
    })

    expect(wrapper.find('.icon').exists()).toBe(true)
  })

  it('renders action button when provided', () => {
    const wrapper = mount(EmptyState, {
      props: {
        title: 'No data',
        actionText: 'Add item',
      },
    })

    expect(wrapper.find('button').exists()).toBe(true)
    expect(wrapper.find('button').text()).toContain('Add item')
  })

  it('emits action event when button clicked', async () => {
    const wrapper = mount(EmptyState, {
      props: {
        title: 'No data',
        actionText: 'Add item',
      },
    })

    await wrapper.find('button').trigger('click')

    expect(wrapper.emitted('action')).toBeTruthy()
  })

  it('does not render button without actionText', () => {
    const wrapper = mount(EmptyState, {
      props: {
        title: 'No data',
      },
    })

    expect(wrapper.find('button').exists()).toBe(false)
  })
})
