import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import SearchInput from '@/components/common/SearchInput.vue'

describe('SearchInput', () => {
  it('renders input field', () => {
    const wrapper = mount(SearchInput)
    expect(wrapper.find('input').exists()).toBe(true)
  })

  it('applies placeholder prop', () => {
    const wrapper = mount(SearchInput, {
      props: { placeholder: 'Search...' },
    })

    expect(wrapper.find('input').attributes('placeholder')).toBe('Search...')
  })

  it('emits search event with debounce', async () => {
    vi.useFakeTimers()

    const wrapper = mount(SearchInput, {
      props: { debounce: 300 },
    })

    await wrapper.find('input').setValue('test query')

    // Should not emit immediately
    expect(wrapper.emitted('search')).toBeFalsy()

    // Advance timers
    vi.advanceTimersByTime(300)

    expect(wrapper.emitted('search')).toBeTruthy()
    expect(wrapper.emitted('search')?.[0]).toEqual(['test query'])

    vi.useRealTimers()
  })

  it('shows clear button when has value', async () => {
    const wrapper = mount(SearchInput, {
      props: { modelValue: 'test' },
    })

    expect(wrapper.find('.clear-btn').exists()).toBe(true)
  })

  it('hides clear button when empty', () => {
    const wrapper = mount(SearchInput, {
      props: { modelValue: '' },
    })

    expect(wrapper.find('.clear-btn').exists()).toBe(false)
  })

  it('clears input when clear button clicked', async () => {
    const wrapper = mount(SearchInput, {
      props: { modelValue: 'test' },
    })

    await wrapper.find('.clear-btn').trigger('click')

    expect(wrapper.emitted('update:modelValue')).toBeTruthy()
    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual([''])
  })

  it('emits input event on keyup', async () => {
    const wrapper = mount(SearchInput)

    await wrapper.find('input').setValue('new value')

    expect(wrapper.emitted('update:modelValue')).toBeTruthy()
  })
})
